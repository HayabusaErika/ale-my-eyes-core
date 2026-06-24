# Ale, My Eyes! - Core

Core library for the Ale, My Eyes! visual assistant application.

## Overview

This crate provides the core functionality for the Ale, My Eyes! application, including:

- **Cloud API Integration**: OpenAI-compatible API support for GPT-4o, Whisper, and TTS
- **Local Inference**: Optional Whisper/ORT ASR, VLM, and LLM support
- **Context Management**: Conversation history, visual memory, and long-term memory
- **Voice Activity Detection**: Real-time speech detection
- **Desktop Automation**: Screen capture, click, type, and scroll actions

## Features

- `cloud` (default): Cloud API integration
- `local-inference`: Local Whisper/ORT ASR, VLM, and LLM support
- `tts`: Text-to-speech support
- `adaptive`: Combined local and cloud inference

## Usage

This crate is used as a git submodule by the PC and Android repositories:

- [ale-my-eyes-pc](https://github.com/risalydevclub/ale-my-eyes-pc) - Desktop application
- [ale-my-eyes-android](https://github.com/risalydevclub/ale-my-eyes-android) - Android application

## License

MIT
