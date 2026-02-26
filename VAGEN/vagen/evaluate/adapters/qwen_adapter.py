"""
Qwen-specific ModelAdapter.

This adapter normalizes Qwen-style responses which often place the
useful human-readable reasoning in `reasoning_content` while leaving
`message.content` empty. It returns a unified envelope string
`<think>...</think><answer>...</answer>` when possible so downstream
code can parse consistently.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import logging
from PIL import Image

from vagen.evaluate.adapters.base_adapter import ModelAdapter
from vagen.evaluate.utils.mm_utils import pil_to_dataurl_png, compile_text_images_for_order

logger = logging.getLogger(__name__)


class QwenAdapter(ModelAdapter):
    """Adapter tuned for Qwen-vl-style responses."""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def _segments_to_content(self, segs: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for kind, val in segs:
            if kind == "text":
                if str(val).strip():
                    content.append({"type": "text", "text": str(val)})
            else:
                content.append({"type": "image_url", "image_url": {"url": pil_to_dataurl_png(val)}})
        return content

    def format_system(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "system", "content": self._segments_to_content(segs)}

    def format_user_turn(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "user", "content": self._segments_to_content(segs)}

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:
        """Call the Qwen-compatible endpoint and normalize the reply.

        Returns a string. Prefer returning a canonical envelope
        `<think>...</think><answer>...</answer>` when possible.
        """
        
        #print("QwenAdapter acompletion called with messages:", messages)
        #print("QwenAdapter acompletion called with chat_config:", chat_config)

        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
            extra_body={
            'enable_thinking': True,
            "thinking_budget": 81920},
        )

        # Log raw response for debugging
        #try:
        #    logger.info("QwenAdapter raw resp repr: %r", completion)
        #except Exception:
        #    logger.info("QwenAdapter raw resp: <unrepresentable>")

        reasoning_content = ""
        answer_content = ""
        is_answering = False

        async for chunk in completion:
            #print("\nReceived chunk:", chunk)
            # 如果chunk.choices为空，则打印usage
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                # 打印思考过程
                reasoning_piece = getattr(delta, "reasoning_content", None)
                if reasoning_piece:
                    #print(reasoning_piece, end='', flush=True)
                    reasoning_content += reasoning_piece
                else:
                    # 开始回复
                    content_piece = delta.content or ""
                    if content_piece != "" and is_answering is False:
                        is_answering = True
                    # 打印回复过程
                    #print(content_piece, end='', flush=True)
                    answer_content += content_piece

        #print("\nFinal reasoning content:", reasoning_content)
        print("Final answer content:", answer_content)

        # Otherwise return content (may be empty)
        return answer_content or ""
