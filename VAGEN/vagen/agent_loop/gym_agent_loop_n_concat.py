# Copyright 2025 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

import asyncio
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from PIL import Image
from .agent_loop_no_concat import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from ..envs.gym_image_env import GymImageEnv
from omegaconf import OmegaConf
import traceback
import importlib
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
from .gym_agent_loop import convert_obs_to_content, extract_success, _flatten_text_only_content, _normalize_images

# -------------------- Configuration --------------------
# Number of recent turns to keep in context (including the current turn).
# e.g., RECENT_N_TURNS=3 means: sys_msg + up to 2 historical (user, assistant) pairs + current user_msg.
# Setting to 1 is equivalent to no-concat mode (no history at all).
RECENT_N_TURNS = 3


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    INTERACTING = "interacting"
    TERMINATED = "terminated"


class AgentData:
    """Container for all mutable trajectory state."""
    def __init__(
        self,
        metrics: Dict[str, Any],
        request_id: str,
        env: GymImageEnv,
        response_limit: int,
        env_name: str,
        sys_msg: Optional[Dict[str, Any]] = None,
        sys_images: Optional[List[Image.Image]] = None,
        cur_msg: Optional[Dict[str, Any]] = None,
        cur_images: Optional[List[Image.Image]] = None,
        group_idx: int = 0,
        traj_idx: int = 0,
    ):
        self.sys_msg: Optional[Dict[str, Any]] = sys_msg
        self.sys_images: Optional[List[Image.Image]] = sys_images
        
        self.cur_msg: Optional[Dict[str, Any]] = cur_msg
        self.cur_images: Optional[List[Image.Image]] = cur_images
        
        # Sliding window of past turns.
        # Each entry: {"user_msg": Dict, "user_images": List[Image], "assistant_text": str}
        self.history: List[Dict[str, Any]] = []
        
        self.metrics = metrics
        self.request_id = request_id
        self.env = env
        self.response_limit = response_limit
        self.env_name = env_name
        self.group_idx = group_idx
        self.traj_idx = traj_idx
        # Token buffers
        self.turn_prompt_ids: Optional[List[int]] = None
        self.turn_response_ids: Optional[List[int]] = None
        self.turn_response_mask: Optional[List[int]] = None
        self.turn_response_logprobs: Optional[List[int]] = None

        # Env stats
        self.env_turns: int = 0

        # Cached assistant text to step env
        self.last_assistant_text: Optional[str] = None
        self.outputs: List[AgentLoopOutput] = []

# -------------------- Gym Agent Loop (N-Concat) --------------------

class GymAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print(f"Performing class-level GymAgentLoop (n-concat, N={RECENT_N_TURNS}) initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.multi_turn_cfg = config.actor_rollout_ref.rollout.multi_turn
        
        cls.env_registry = {}

        for k, v in config.env_registry.items():
            module_path, class_name = v.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls.env_registry[k] = getattr(module, class_name)
            
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

    @staticmethod
    def _build_windowed_messages_and_images(agent_data: AgentData):
        """Build messages list and image list from the sliding window of recent turns.
        
        Returns:
            messages: [sys_msg, hist_user_0, hist_asst_0, ..., cur_msg]
            all_images: concatenated images from sys + history + current
        """
        messages = [agent_data.sys_msg]
        all_images = list(agent_data.sys_images)
        
        for entry in agent_data.history:
            messages.append(entry["user_msg"])
            all_images.extend(entry["user_images"])
            messages.append({"role": "assistant", "content": entry["assistant_text"]})
        
        messages.append(agent_data.cur_msg)
        all_images.extend(agent_data.cur_images)
        
        return messages, all_images

    @rollout_trace_op
    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        metrics: Dict[str, Any] = {}
        request_id = uuid4().hex

        # Build env
        env_cls = self.env_registry[kwargs["env_name"]]
        env_config = kwargs["config"]
        seed = kwargs["seed"]
        self.env_max_turns = kwargs.get("max_turns", None)
        env: GymImageEnv = env_cls(env_config=env_config)

        # Bootstrap: reset -> system_prompt (message order: system, then initial user)
        init_obs, info = await env.reset(seed=seed)
        sys_obs = await env.system_prompt()

        sys_msg={"role": "system", "content": convert_obs_to_content(sys_obs, **kwargs)}
        sys_images=_normalize_images(sys_obs.get("multi_modal_input", {}).get("<image>", []) or [])
        
        cur_msg={"role": "user", "content": convert_obs_to_content(init_obs, **kwargs)}
        cur_images=_normalize_images(init_obs.get("multi_modal_input", {}).get("<image>", []) or [])

        per_turn_response_limit = int(kwargs.get("response_length_per_turn") or self.response_length)
        per_turn_response_limit = min(per_turn_response_limit, self.response_length)
        if per_turn_response_limit <= 0:
            per_turn_response_limit = 1

        agent_data = AgentData(
            sys_msg=sys_msg,
            sys_images=sys_images,
            cur_msg=cur_msg,
            cur_images=cur_images,
            metrics=metrics,
            request_id=request_id,
            env=env,
            response_limit=per_turn_response_limit,
            env_name=kwargs["env_name"],
            group_idx=kwargs.get("group_idx", 0),
            traj_idx=kwargs.get("traj_idx", 0),
        )

        # State machine: always GENERATE -> INTERACT, and decide termination inside INTERACT
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.INTERACTING:
                state = await self._handle_env_state(agent_data, **kwargs)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Close env after loop
        await env.close()

        # Verl's AgentLoopManager expects a single AgentLoopOutput, not a list.
        # Merge multiple turn outputs into one (concat style) when needed.
        outputs = agent_data.outputs
        if len(outputs) == 0:
            raise RuntimeError("gym_agent_loop_n_concat: no outputs produced")
        if len(outputs) == 1:
            return outputs[0]
        # Merge: prompt=first.prompt, response=resp_0 + prompt_1 + resp_1 + ... (interleaved, prompt parts masked 0)
        merged_prompt_ids = outputs[0].prompt_ids
        merged_response_ids: List[int] = []
        merged_response_mask: List[int] = []
        merged_response_logprobs: Optional[List[float]] = None
        merged_images: List[Any] = []
        for j, o in enumerate(outputs):
            merged_response_ids.extend(o.response_ids)
            merged_response_mask.extend(o.response_mask)
            if o.response_logprobs is not None:
                if merged_response_logprobs is None:
                    merged_response_logprobs = []
                merged_response_logprobs.extend(o.response_logprobs)
            if o.multi_modal_data and "image" in o.multi_modal_data:
                merged_images.extend(o.multi_modal_data["image"])
            if j < len(outputs) - 1:
                next_o = outputs[j + 1]
                merged_response_ids.extend(next_o.prompt_ids)
                merged_response_mask.extend([0] * len(next_o.prompt_ids))
                if merged_response_logprobs is not None and next_o.response_logprobs is not None:
                    merged_response_logprobs.extend([0.0] * len(next_o.prompt_ids))
        last = outputs[-1]
        merged_mm = {"image": merged_images} if merged_images else (last.multi_modal_data or {})
        return AgentLoopOutput(
            prompt_ids=merged_prompt_ids[-self.prompt_length:] if len(merged_prompt_ids) > self.prompt_length else merged_prompt_ids,
            response_ids=merged_response_ids[: self.response_length],
            response_mask=merged_response_mask[: self.response_length],
            multi_modal_data=merged_mm,
            response_logprobs=merged_response_logprobs[: self.response_length] if merged_response_logprobs else None,
            reward_score=last.reward_score,
            num_turns=sum(getattr(o, "num_turns", 1) for o in outputs),
            metrics=last.metrics,
            extra_fields=last.extra_fields or {},
        )

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: Dict[str, Any]) -> AgentState:
        """Encode windowed messages (sys + recent history + current user) into turn_prompt_ids."""
        messages, all_images = self._build_windowed_messages_and_images(agent_data)

        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=all_images or None, return_tensors="pt")
            agent_data.turn_prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if all_images:
                raise ValueError("Environment returned images but `processor` is None.")
            try:
                agent_data.turn_prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )
            except TypeError as e:
                logger.warning(f"TypeError Warning in apply_chat_template: {e}, switching to flattened text-only content.")
                flat_messages = [_flatten_text_only_content(msg) for msg in messages]
                agent_data.turn_prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        flat_messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )
        
        if len(agent_data.turn_prompt_ids)>self.prompt_length:
            logger.warning(f"In env:{agent_data.env_name}, prompt length {len(agent_data.turn_prompt_ids)} exceeds prompt_length {self.prompt_length}")
        return AgentState.GENERATING

    
    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: Dict[str, Any]
    ) -> AgentState:
        """Generate assistant output and mark generated tokens with mask=1."""
        sampling_params_for_turn = sampling_params.copy()
        max_new_tokens=sampling_params_for_turn.get("max_new_tokens", None) or agent_data.response_limit
        max_new_tokens = min(max_new_tokens, agent_data.response_limit)
        sampling_params_for_turn["max_new_tokens"] = max_new_tokens
        # Use all images from the windowed context
        _, all_images = self._build_windowed_messages_and_images(agent_data)

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id = agent_data.request_id,
                prompt_ids = agent_data.turn_prompt_ids,
                sampling_params = sampling_params_for_turn,
                image_data = all_images,
            )


        agent_data.turn_response_ids = output.token_ids
        agent_data.turn_response_mask = [1] * len(output.token_ids)
        agent_data.turn_prompt_ids += agent_data.turn_response_ids
        if output.log_probs:
            agent_data.turn_response_logprobs = output.log_probs

        # Cache assistant text and add assistant message (text-only)
        assistant_message = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.turn_response_ids, skip_special_tokens=True)
        )
        agent_data.last_assistant_text = assistant_message
        return AgentState.INTERACTING

    async def _handle_env_state(self, agent_data: AgentData, **kwargs) -> AgentState:
        """
        Step the environment with last assistant action; always collect reward first.
        If terminal (done/success/turn-limit/token-limit), stop WITHOUT appending user suffix,
        so the episode ends on an assistant turn.
        """
        action_str = agent_data.last_assistant_text or ""
        try:
            obs, reward, done, info = await agent_data.env.step(action_str)
            # traceback
        except Exception as exc:
            logger.error(
                "Environment step failed in '%s' with action %r: %s",
                agent_data.env_name,
                action_str,
                exc,
            )
            logger.error("Environment traceback:\n%s", traceback.format_exc())
            obs, reward, done, info = {"obs_str":"Environment Error"}, 0.0, True, {"traj_success": False}

        traj_success = extract_success(info)
        agent_data.env_turns += 1
        last_turn=False
        
        if done:
            last_turn = True

        if self.env_max_turns is not None and agent_data.env_turns >= int(self.env_max_turns):
            last_turn = True

        if len(agent_data.turn_response_mask) >= self.response_length:
            last_turn = True

        _, turn_images = self._build_windowed_messages_and_images(agent_data)
        
        resp_len = len(agent_data.turn_response_mask)
        response_ids = agent_data.turn_prompt_ids[-resp_len:] if resp_len else []
        prompt_ids = agent_data.turn_prompt_ids[: len(agent_data.turn_prompt_ids) - resp_len]
        multi_modal_data = {"image": turn_images} if turn_images else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids[-self.prompt_length:],
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.turn_response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=(
                agent_data.turn_response_logprobs[: self.response_length] if agent_data.turn_response_logprobs else None
            ),
            reward_score=float(reward),
            num_turns=1,
            metrics=agent_data.metrics,
            extra_fields={"reward_extra_info": {
                "traj_success": float(traj_success)},
                "image_data": turn_images,
                "last_turn": last_turn,
                "group_idx": agent_data.group_idx,
                "traj_idx": agent_data.traj_idx,
                "turn_idx": agent_data.env_turns,
                          
            },
        )
        agent_data.outputs.append(output)
        
        # ---- N-concat: save completed turn to history, then trim ----
        agent_data.history.append({
            "user_msg": agent_data.cur_msg,
            "user_images": list(agent_data.cur_images),
            "assistant_text": agent_data.last_assistant_text,
        })
        # Keep only the last (RECENT_N_TURNS - 1) entries;
        # the current (new) turn will be the N-th when we re-enter PENDING.
        if len(agent_data.history) > RECENT_N_TURNS - 1:
            agent_data.history = agent_data.history[-(RECENT_N_TURNS - 1):]

        # Update current message and images for next turn
        cur_msg={"role": "user", "content": convert_obs_to_content(obs, **kwargs)}
        cur_images=_normalize_images(obs.get("multi_modal_input", {}).get("<image>", []) or [])
        agent_data.cur_msg = cur_msg
        agent_data.cur_images = cur_images
        if last_turn:
            return AgentState.TERMINATED

        return AgentState.PENDING