#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import argparse
import asyncio
import os
import sys
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()

from call_connection_manager import CallConfigManager, SessionManager
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMMessagesFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

class TranslationProcessor(FrameProcessor):
    """A processor that translates text frames from a source language to a target language."""

    def __init__(self, in_language, out_language):
        """Initialize the TranslationProcessor with source and target languages.

        Args:
            in_language (str): The language of the input text.
            out_language (str): The language to translate the text into.
        """
        super().__init__()
        self._out_language = out_language
        self._in_language = in_language

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame and translate text frames.

        Args:
            frame (Frame): The frame to process.
            direction (FrameDirection): The direction of the frame.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            logger.debug(f"Translating {self._in_language}: {frame.text} to {self._out_language}")
            context = [
                {
                    "role": "system",
                    "content": f"You will be provided with a sentence in {self._in_language}, and your task is to only translate it into {self._out_language}.",
                },
                {"role": "user", "content": frame.text},
            ]
            await self.push_frame(LLMMessagesFrame(context))
        else:
            await self.push_frame(frame)


async def main(
    room_url: str,
    token: str,
    body: dict,
):
    # ------------ CONFIGURATION AND SETUP ------------

    # Create a config manager using the provided body
    call_config_manager = CallConfigManager.from_json_string(body) if body else CallConfigManager()

    # Get dialin settings if present
    dialin_settings = call_config_manager.get_dialin_settings()

    # Initialize the session manager
    session_manager = SessionManager()

    # ------------ TRANSPORT SETUP ------------

    daily_dialin_settings = DailyDialinSettings(
        call_id=dialin_settings.get("call_id"), call_domain=dialin_settings.get("call_domain")
    )
    transport_params = DailyParams(
        api_url=daily_api_url,
        api_key=daily_api_key,
        dialin_settings=daily_dialin_settings,
        audio_in_enabled=True,
        audio_out_enabled=True,
        camera_out_enabled=False,
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
        vad_audio_passthrough=True,
        transcription_enabled=False,
    )

    # Initialize transport with Daily
    transport = DailyTransport(
        room_url,
        token,
        "Simple Dial-in Bot",
        transport_params,
    )

    # Initialize STT
    stt = OpenAISTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini-transcribe",
    )

    # Initialize TTS
    tts = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini-tts",
    )

    # ------------ LLM AND CONTEXT SETUP ------------

    # Initialize LLM
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini-2024-07-18")

    # Initialize LLM context and aggregator
    # context = OpenAILLMContext()
    # context_aggregator = llm.create_context_aggregator(context)

    # ------------ PIPELINE SETUP ------------

    in_language = "English"
    out_language = "Spanish"

    tp = TranslationProcessor(in_language=in_language, out_language=out_language)

    rtvi = RTVIProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            tp,
            llm,
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=False,  # We don't want to interrupt the translator bot
        ),
        observers=[RTVIObserver(rtvi)],
    )
    
    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        logger.debug(f"participant joined: {participant['id']}")

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.debug(f"Participant left: {participant}, reason: {reason}")
        await task.cancel()

    # ------------ RUN PIPELINE ------------

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Dial-in Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()

    # Log the arguments for debugging
    logger.info(f"Room URL: {args.url}")
    logger.info(f"Token: {args.token}")
    logger.info(f"Body provided: {bool(args.body)}")

    asyncio.run(main(args.url, args.token, args.body))
