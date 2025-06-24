"""
title: Example Filter
author: open-webui
version: 0.1
"""

from pydantic import BaseModel, Field
from typing import Optional, Callable, Any, Awaitable


class Filter:

    def __init__(self):

        pass

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> dict:
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Ce produit est en phase beta. Comme toute IA, il peut faire des erreurs. Pensez à vérifier les réponses données.",
                    "done": True,
                },
            }
        )

        return body
