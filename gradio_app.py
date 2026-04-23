#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import shutil
from functools import partial
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download

try:
    from mutagen.wave import WAVE
    from mutagen.id3 import COMM, TIT2, TXXX, Encoding

    _MUTAGEN_AVAILABLE = True
except ImportError:
    WAVE = None
    COMM = None
    TIT2 = None
    TXXX = None
    Encoding = None
    _MUTAGEN_AVAILABLE = False

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    default_runtime_device,
    get_cached_runtime,
    list_available_runtime_devices,
    list_available_runtime_precisions,
    save_wav,
)

FIXED_SECONDS = 30.0
MAX_GRADIO_CANDIDATES = 32
GRADIO_AUDIO_COLS_PER_ROW = 8
PROMPT_BOX_ID = "prompt_box"
EMOJI_PANEL_ID = "emoji-panel"
TEXT_EMOJI_ROW_ID = "text-emoji-row"
PROMPT_COLUMN_ID = "prompt-column"
EMOJI_COLUMN_ID = "emoji-column"
EMOJI_HTML_ID = "emoji-panel-html"
PROMPT_ACTIONS_ROW_ID = "prompt-actions-row"
PROMPT_GENERATE_BUTTON_ID = "prompt-generate-button"
PROMPT_CLEAR_BUTTON_ID = "prompt-clear-button"
MAIN_GENERATE_BUTTON_ID = "main-generate-button"
SEED_CLEAR_BUTTON_ID = "seed-clear-button"
SEED_LAST_BUTTON_ID = "seed-last-button"
SEED_TEXTBOX_ID = "seed-textbox"
SAVE_GENERATED_AUDIO_BUTTON_CLASS = "save-generated-audio-button"
SETTINGS_FILE_NAME = "gradio_app_settings.json"
DEFAULT_OUTPUT_DIR = "output_voice"
APP_DATA_ROOT = Path(__file__).resolve().parent
REFERENCE_PRESET_ROWS = 3
REFERENCE_PRESET_COLS = 6
REFERENCE_PRESET_COUNT = REFERENCE_PRESET_ROWS * REFERENCE_PRESET_COLS
REFERENCE_PRESET_DIR_NAME = "reference_audio_presets"
REFERENCE_PRESET_BUTTON_ID_PREFIX = "reference-preset-button-"
REFERENCE_PRESET_UPLOAD_CLASS = "reference-preset-upload"
REFERENCE_PRESET_SET_BUTTON_CLASS = "reference-preset-set-button"
REFERENCE_PRESET_CLEAR_BUTTON_CLASS = "reference-preset-clear-button"
REFERENCE_PRESET_COLOR_BUTTON_CLASS = "reference-preset-color-button"
REFERENCE_PRESET_STYLE_HTML_ID = "reference-preset-style"
REFERENCE_PRESET_NOTICE_ID = "reference-preset-notice"
REFERENCE_PRESET_COLOR_INPUT_ID_PREFIX = "reference-preset-color-input-"
REFERENCE_PRESET_PATH_INPUT_ID_PREFIX = "reference-preset-path-input-"
REFERENCE_PRESET_COLOR_BUTTON_ID_PREFIX = "reference-preset-color-button-"
REFERENCE_PRESET_SLOT_ID_PREFIX = "reference-preset-slot-"
REFERENCE_PRESET_REORDER_INPUT_ID = "reference-preset-reorder-input"
PRESET_METADATA_DROP_INPUT_ID = "preset-metadata-drop-input"
DEFAULT_REFERENCE_PRESET_COLOR = "#2f9e44"
DEFAULT_REFERENCE_PRESET_ASSIGNED_COLOR = "#1971c2"
REFERENCE_PRESET_LABEL_MAX_CHARS = 20
REFERENCE_PRESET_PALETTE_COLORS = [
    "#111111",
    "#6b7280",
    "#c92a2a",
    "#e67700",
    "#f2c94c",
    "#808000",
    "#8d6e63",
    "#2f9e44",
    "#0b8f87",
    "#0f52ba",
    "#8e44ad",
    "#d63384",
]

EMOJI_GROUPS: list[tuple[str, list[str]]] = [
    ("Emotions", ["😏", "🥺", "😭", "😪", "😰", "😆", "😠", "😲", "😖", "😟", "🫣", "🙄", "😊", "🥴", "😌", "🤔"]),
    ("Breath, Mouth, Throat", ["😮‍💨", "🌬️", "😮", "👅", "💋", "🥤", "🤧", "😒", "🥱", "🥵"]),
    ("Reactions, Vocalization", ["🙏", "👂", "🤭", "👌", "🎵", "😱", "🤐", "🫶"]),
    ("Effects, Speed", ["📢", "📞", "⏸️", "⏩", "🐢"]),
]

EMOJI_TOOLTIPS: dict[str, str] = {
    "👂": "囁き、耳元の音",
    "😮‍💨": "吐息、溜息、寝息",
    "⏸️": "間、沈黙",
    "🤭": "笑い（くすくす、含み笑いなど）",
    "🥵": "喘ぎ、うめき声、唸り声",
    "📢": "エコー、リバーブ",
    "😏": "からかうように、甘えるように",
    "🥺": "声を震わせながら、自信のなさげに",
    "🌬️": "息切れ、荒い息遣い、呼吸音",
    "😮": "息をのむ",
    "👅": "舐める音、咀嚼音、水音",
    "💋": "リップノイズ",
    "🫶": "優しく",
    "😭": "嗚咽、泣き声、悲しみ",
    "😱": "悲鳴、叫び、絶叫",
    "😪": "眠そうに、気だるげに",
    "⏩": "早口、一気にまくしたてる、急いで",
    "📞": "電話越し、スピーカー越しのような音",
    "🐢": "ゆっくりと",
    "🥤": "唾を飲み込む音",
    "🤧": "咳き込み、鼻をすする、くしゃみ、咳払い",
    "😒": "舌打ち",
    "😰": "慌てて、動揺、緊張、どもり",
    "😆": "喜びながら",
    "😠": "怒り、不満げに、拗ねながら",
    "😲": "驚き、感嘆",
    "🥱": "あくび",
    "😖": "苦しげに",
    "😟": "心配そうに",
    "🫣": "恥ずかしそうに、照れながら",
    "🙄": "呆れたように",
    "😊": "楽しげに、嬉しそうに",
    "👌": "相槌、頷く音",
    "🙏": "懇願するように",
    "🥴": "酔っ払って",
    "🎵": "鼻歌",
    "🤐": "口を塞がれて",
    "😌": "安堵、満足げに",
    "🤔": "疑問の声",
}

CUSTOM_CSS = f"""
#{TEXT_EMOJI_ROW_ID} {{
  align-items: flex-start;
  gap: 12px;
}}
#{PROMPT_COLUMN_ID} {{
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-top: 10px;
}}
#{PROMPT_ACTIONS_ROW_ID} {{
  display: flex !important;
  justify-content: flex-end;
  min-height: 32px;
  flex: 0 0 auto;
}}
#{PROMPT_ACTIONS_ROW_ID} > div {{
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
}}
#{PROMPT_CLEAR_BUTTON_ID} {{
  align-self: flex-end;
  width: auto !important;
  min-width: 0 !important;
  flex: 0 0 auto !important;
}}
#{PROMPT_CLEAR_BUTTON_ID} button {{
  width: auto !important;
  min-width: 0 !important;
  min-height: 32px !important;
  height: 32px !important;
  padding: 0 8px !important;
  font-size: 0.72rem !important;
  white-space: nowrap !important;
  line-height: 1 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}}
#{EMOJI_COLUMN_ID} {{
  display: flex;
  flex-direction: column;
  min-height: 360px;
}}
#{PROMPT_BOX_ID} {{
  flex: 0 0 auto;
}}
#{PROMPT_BOX_ID} textarea {{
  min-height: 155px !important;
}}
#{PROMPT_GENERATE_BUTTON_ID} button,
#{MAIN_GENERATE_BUTTON_ID} button {{
  width: 100%;
  min-height: 44px !important;
}}
#{EMOJI_HTML_ID} {{
  flex: 1 1 auto;
  min-height: 0;
}}
#{EMOJI_PANEL_ID} {{
  border: 1px solid var(--border-color-primary);
  border-radius: 12px;
  padding: 8px 10px;
  background: var(--block-background-fill);
  height: 100%;
  overflow-y: auto;
}}
#{EMOJI_PANEL_ID} .emoji-section + .emoji-section {{
  margin-top: 8px;
}}
#{EMOJI_PANEL_ID} .emoji-title {{
  font-size: 0.84rem;
  font-weight: 600;
  margin-bottom: 4px;
  color: var(--body-text-color);
}}
#{EMOJI_PANEL_ID} .emoji-grid {{
  display: grid;
  grid-template-columns: repeat(8, minmax(0, 1fr));
  gap: 5px;
}}
#{EMOJI_PANEL_ID} .emoji-btn {{
  width: 100%;
  min-width: 0;
  height: 1.95rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  border: 1px solid var(--border-color-primary);
  border-radius: 9px;
  background: var(--button-secondary-background-fill);
  font-size: 0.95rem;
  line-height: 1;
  text-align: center;
  cursor: pointer;
  transition: transform 0.10s ease, background-color 0.10s ease, border-color 0.10s ease, box-shadow 0.10s ease;
}}
#{EMOJI_PANEL_ID} .emoji-btn:hover,
#{EMOJI_PANEL_ID} .emoji-btn:focus-visible {{
  background: var(--button-secondary-background-fill-hover);
  border-color: var(--color-accent-soft);
  box-shadow: 0 0 0 1px var(--color-accent-soft);
  transform: scale(1.015);
  outline: none;
}}
#seed-input-wrap {{
  display: flex;
  flex-direction: column;
  gap: 6px;
}}
#seed-label {{
  font-size: var(--block-label-text-size);
  font-weight: var(--block-label-text-weight);
  line-height: var(--line-sm);
  color: var(--block-label-text-color);
  margin-bottom: 0;
}}
#seed-input-row {{
  display: flex !important;
  flex-wrap: nowrap !important;
  gap: 0 !important;
  align-items: stretch !important;
}}
#seed-input-row > div:first-child {{
  flex: 1 1 auto !important;
  min-width: 0 !important;
}}
#seed-input-row > div:nth-child(2),
#seed-input-row > div:nth-child(3) {{
  flex: 0 0 56px !important;
  width: 56px !important;
  min-width: 56px !important;
}}
#{SEED_TEXTBOX_ID} {{
  min-width: 0 !important;
}}
#{SEED_TEXTBOX_ID} textarea,
#{SEED_TEXTBOX_ID} input {{
  min-height: 40px !important;
  height: 40px !important;
  line-height: 1.25 !important;
  padding-top: 8px !important;
  padding-bottom: 8px !important;
  border-top-right-radius: 0 !important;
  border-bottom-right-radius: 0 !important;
  box-sizing: border-box !important;
}}
#{SEED_CLEAR_BUTTON_ID},
#{SEED_LAST_BUTTON_ID} {{
  min-width: 56px !important;
  width: 56px !important;
}}
#{SEED_CLEAR_BUTTON_ID} button,
#{SEED_LAST_BUTTON_ID} button {{
  width: 56px !important;
  min-width: 56px !important;
  height: 40px !important;
  min-height: 40px !important;
  padding: 0 10px !important;
  border: 1px solid var(--border-color-primary);
  border-left: none;
  border-top-left-radius: 0 !important;
  border-bottom-left-radius: 0 !important;
  border-top-right-radius: 8px !important;
  border-bottom-right-radius: 8px !important;
  box-shadow: none !important;
  font-size: 0.82rem !important;
}}
.{SAVE_GENERATED_AUDIO_BUTTON_CLASS} {{
  min-height: 28px !important;
  height: 28px !important;
  padding: 0 10px !important;
  font-size: 0.78rem !important;
  background: #2d6cdf !important;
  border: 1px solid #255ec7 !important;
  color: #ffffff !important;
  box-shadow: none !important;
  transition: background-color 0.15s ease, transform 0.06s ease, box-shadow 0.15s ease !important;
}}
.{SAVE_GENERATED_AUDIO_BUTTON_CLASS}:hover {{
  background: #255fc6 !important;
}}
.{SAVE_GENERATED_AUDIO_BUTTON_CLASS}:active {{
  background: #1f4fa4 !important;
  transform: translateY(1px);
}}
.{SAVE_GENERATED_AUDIO_BUTTON_CLASS}[disabled] {{
  background: #1f4fa4 !important;
  border-color: #1a4389 !important;
  color: #ffffff !important;
  opacity: 1 !important;
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.08) inset !important;
}}
#{REFERENCE_PRESET_STYLE_HTML_ID} {{
  min-height: 0 !important;
}}
.reference-preset-slot {{
  gap: 0 !important;
  padding: 8px;
  margin: 4px 3px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.035), rgba(255, 255, 255, 0.015));
  box-shadow: 0 6px 14px rgba(0, 0, 0, 0.14);
  overflow: visible !important;
  position: relative;
  z-index: 1;
  cursor: grab;
}}
.reference-preset-slot[data-dragging="true"] {{
  opacity: 0.62;
  cursor: grabbing;
}}
.reference-preset-slot[data-drop-target="true"] {{
  outline: 2px dashed rgba(79, 138, 255, 0.9);
  outline-offset: -2px;
  box-shadow: 0 0 0 2px rgba(79, 138, 255, 0.16), 0 6px 14px rgba(0, 0, 0, 0.14);
}}
.reference-preset-drag-preview {{
  position: fixed;
  left: 0;
  top: 0;
  z-index: 9999;
  pointer-events: none;
  width: 118px;
  padding: 8px 9px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  background: linear-gradient(180deg, rgba(34, 39, 46, 0.96), rgba(20, 24, 30, 0.96));
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.32);
  transform: translate(14px, 12px);
  opacity: 0.96;
}}
.reference-preset-drag-preview .preview-label {{
  display: block;
  font-size: 0.68rem;
  line-height: 1.1;
  font-weight: 700;
  color: #f5f7fb;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.reference-preset-drag-preview .preview-sub {{
  display: block;
  margin-top: 4px;
  font-size: 0.56rem;
  line-height: 1;
  color: rgba(245, 247, 251, 0.72);
}}
.{REFERENCE_PRESET_UPLOAD_CLASS} {{
  min-height: 48px !important;
  position: relative !important;
  overflow: hidden !important;
  margin-bottom: 0 !important;
  padding-bottom: 0 !important;
}}
.{REFERENCE_PRESET_UPLOAD_CLASS} label {{
  font-size: 0.64rem !important;
  line-height: 1.05 !important;
}}
.{REFERENCE_PRESET_UPLOAD_CLASS} .file-wrap,
.{REFERENCE_PRESET_UPLOAD_CLASS} .wrap,
.{REFERENCE_PRESET_UPLOAD_CLASS} .center,
.{REFERENCE_PRESET_UPLOAD_CLASS} .file-preview,
.{REFERENCE_PRESET_UPLOAD_CLASS} .file-upload {{
  min-height: 48px !important;
  margin-bottom: 0 !important;
  padding-bottom: 0 !important;
}}
.{REFERENCE_PRESET_UPLOAD_CLASS} .file-upload {{
  padding: 2px 6px !important;
  font-size: 0 !important;
}}
.{REFERENCE_PRESET_UPLOAD_CLASS} .center {{
  font-size: 0 !important;
  line-height: 0 !important;
  color: transparent !important;
  cursor: default !important;
}}
.{REFERENCE_PRESET_UPLOAD_CLASS} .center * {{
  font-size: 0 !important;
  line-height: 0 !important;
  color: transparent !important;
  min-height: 0 !important;
}}
.{REFERENCE_PRESET_UPLOAD_CLASS} button,
.{REFERENCE_PRESET_UPLOAD_CLASS} input[type="file"],
.{REFERENCE_PRESET_UPLOAD_CLASS} label {{
  cursor: default !important;
}}
.{REFERENCE_PRESET_UPLOAD_CLASS} .center-text,
.{REFERENCE_PRESET_UPLOAD_CLASS} .or,
.{REFERENCE_PRESET_UPLOAD_CLASS} .upload-text,
.{REFERENCE_PRESET_UPLOAD_CLASS} .file-upload-text,
.{REFERENCE_PRESET_UPLOAD_CLASS} .hint,
.{REFERENCE_PRESET_UPLOAD_CLASS} .file-legend,
.{REFERENCE_PRESET_UPLOAD_CLASS} .file-description {{
  display: none !important;
}}
.{REFERENCE_PRESET_UPLOAD_CLASS} .file-preview,
.{REFERENCE_PRESET_UPLOAD_CLASS} .file-preview-item,
.{REFERENCE_PRESET_UPLOAD_CLASS} .file-info,
.{REFERENCE_PRESET_UPLOAD_CLASS} .file-name,
.{REFERENCE_PRESET_UPLOAD_CLASS} .file-size,
.{REFERENCE_PRESET_UPLOAD_CLASS} .file-meta,
.{REFERENCE_PRESET_UPLOAD_CLASS} [data-testid="file-preview"],
.{REFERENCE_PRESET_UPLOAD_CLASS} [data-testid="file-preview-item"] {{
  font-size: 0.52rem !important;
  line-height: 1.0 !important;
  min-height: 0 !important;
  max-height: 18px !important;
}}
.{REFERENCE_PRESET_UPLOAD_CLASS} .file-preview,
.{REFERENCE_PRESET_UPLOAD_CLASS} [data-testid="file-preview"] {{
  overflow: hidden !important;
}}
.reference-preset-prompt-preview {{
  display: none !important;
}}
.reference-preset-upload-inline-preview {{
  position: absolute;
  left: 9px;
  right: 9px;
  top: 28px;
  overflow: hidden;
  pointer-events: none;
  color: #8b8f97;
  font-size: 0.6rem;
  line-height: 1.18;
  padding-top: 1px;
  white-space: nowrap;
  text-overflow: ellipsis;
  z-index: 2;
}}
.reference-preset-upload-inline-preview.is-empty {{
  display: none;
}}
.reference-preset-button-row {{
  margin-top: -5px !important;
  gap: 3px;
  align-items: flex-start;
  flex-wrap: nowrap !important;
  overflow: visible !important;
}}
 .reference-preset-button-row > * {{
  min-width: 0 !important;
}}
.reference-preset-button-tools {{
  display: flex;
  flex: 0 0 18px !important;
  width: 18px !important;
  min-width: 18px !important;
  flex-direction: column;
  gap: 2px;
  align-items: stretch;
  overflow: visible !important;
}}
.reference-preset-button-tools > .reference-preset-color-host {{
  flex: 0 0 16px !important;
  width: 18px !important;
  min-width: 18px !important;
  min-height: 16px !important;
  height: 16px !important;
  margin: 0 !important;
  padding: 0 !important;
  line-height: 0 !important;
  overflow: visible !important;
}}
.reference-preset-button-tools > .reference-preset-color-host > div {{
  margin: 0 !important;
  padding: 0 !important;
}}
.{REFERENCE_PRESET_SET_BUTTON_CLASS} {{
  flex: 1 1 auto !important;
  width: auto !important;
  min-width: 0 !important;
  min-height: 34px !important;
  height: auto !important;
  padding: 2px 3px !important;
  font-size: 0.64rem !important;
  font-weight: 700 !important;
  line-height: 0.95 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  border-radius: 10px !important;
  border-width: 1px !important;
  box-shadow: 0 3px 8px rgba(0, 0, 0, 0.22), 0 1px 0 rgba(255, 255, 255, 0.18) inset !important;
  transition: background-color 0.15s ease, transform 0.06s ease, box-shadow 0.15s ease, filter 0.15s ease, border-color 0.15s ease !important;
  cursor: grab !important;
}}
.{REFERENCE_PRESET_SET_BUTTON_CLASS}:hover {{
  filter: brightness(1.04) !important;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.26), 0 1px 0 rgba(255, 255, 255, 0.2) inset !important;
}}
.{REFERENCE_PRESET_SET_BUTTON_CLASS}:active {{
  transform: translateY(1px);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.22), 0 1px 0 rgba(255, 255, 255, 0.12) inset !important;
  cursor: grabbing !important;
}}
.{REFERENCE_PRESET_SET_BUTTON_CLASS}[disabled] {{
  opacity: 1 !important;
  pointer-events: none !important;
  cursor: default !important;
}}
.{REFERENCE_PRESET_SET_BUTTON_CLASS} span,
.{REFERENCE_PRESET_SET_BUTTON_CLASS} .wrap,
.{REFERENCE_PRESET_SET_BUTTON_CLASS} .svelte-1ipelgc {{
  font-size: 0.64rem !important;
  line-height: 0.95 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}}
.{REFERENCE_PRESET_CLEAR_BUTTON_CLASS} {{
  flex: 0 0 18px !important;
  width: 18px !important;
  min-width: 18px !important;
  min-height: 16px !important;
  height: 16px !important;
  padding: 0 !important;
  border-radius: 6px !important;
  font-size: 0.58rem !important;
  font-weight: 700 !important;
  background: #d9dde3 !important;
  border: 1px solid #c4c9d1 !important;
  color: #5f6773 !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.10) !important;
}}
.{REFERENCE_PRESET_CLEAR_BUTTON_CLASS}:hover {{
  background: #cfd5dd !important;
}}
.{REFERENCE_PRESET_CLEAR_BUTTON_CLASS}:active {{
  background: #c2c8d1 !important;
  transform: translateY(1px);
}}
.{REFERENCE_PRESET_CLEAR_BUTTON_CLASS}[disabled] {{
  opacity: 0.55 !important;
}}
.{REFERENCE_PRESET_COLOR_BUTTON_CLASS} {{
  flex: 0 0 16px !important;
  width: 18px !important;
  min-width: 18px !important;
  min-height: 16px !important;
  height: 16px !important;
  padding: 0 !important;
  border-radius: 6px !important;
  font-size: 0.56rem !important;
  font-weight: 700 !important;
  background: #eef1f4 !important;
  border: 1px solid #ccd2d9 !important;
  color: #58606c !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08) !important;
}}
.reference-preset-color-widget {{
  position: relative;
  width: 18px;
  min-width: 18px;
  height: 16px;
  overflow: visible;
}}
.reference-preset-color-trigger {{
  display: block;
  width: 18px;
  height: 16px;
  margin: 0;
  padding: 0;
  border-radius: 6px;
  border: 1px solid rgba(0, 0, 0, 0.18);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
  cursor: pointer;
}}
.reference-preset-color-trigger[disabled] {{
  opacity: 0.55;
  cursor: default;
}}
.reference-preset-color-palette {{
  position: absolute;
  right: 0;
  bottom: 18px;
  top: auto;
  z-index: 40;
  display: none;
  grid-template-columns: repeat(6, 12px);
  gap: 3px;
  padding: 5px;
  border-radius: 8px;
  background: rgba(20, 24, 30, 0.96);
  border: 1px solid rgba(255, 255, 255, 0.12);
  box-shadow: 0 8px 18px rgba(0, 0, 0, 0.28);
}}
.reference-preset-color-palette[data-open="true"] {{
  display: grid;
}}
.reference-preset-color-swatch {{
  width: 12px;
  height: 12px;
  border: 1px solid rgba(255, 255, 255, 0.35);
  border-radius: 3px;
  padding: 0;
  cursor: pointer;
}}
.reference-preset-color-swatch[data-selected="true"] {{
  outline: 1px solid #ffffff;
  outline-offset: 1px;
}}
.reference-preset-color-value {{
  display: none !important;
}}
.reference-preset-path-value {{
  display: none !important;
}}
.reference-preset-reorder-value {{
  display: none !important;
}}
.preset-metadata-drop-value {{
  display: none !important;
}}
#{PROMPT_BOX_ID}[data-preset-drop-target="true"] {{
  outline: 2px dashed rgba(79, 138, 255, 0.9);
  outline-offset: 2px;
  border-radius: 12px;
}}
#{REFERENCE_PRESET_NOTICE_ID} {{
  min-height: 1.2rem;
  margin-top: 4px;
}}
#{REFERENCE_PRESET_NOTICE_ID} .preset-notice {{
  font-size: 0.84rem;
  line-height: 1.25;
  color: #c92a2a;
  font-weight: 600;
}}
"""

CUSTOM_HEAD = f"""
<script>
(() => {{
  const promptBoxId = {PROMPT_BOX_ID!r};
  const promptSelector = `#${{promptBoxId}} textarea`;
  const state = window.__emojiPromptState || {{ start: null, end: null }};
  window.__emojiPromptState = state;

  function getPromptTextarea() {{
    const root = document.getElementById(promptBoxId);
    return root ? root.querySelector('textarea') : null;
  }}

  function rememberSelection(textarea) {{
    const target = textarea || getPromptTextarea();
    if (!target) return;
    state.start = typeof target.selectionStart === 'number' ? target.selectionStart : target.value.length;
    state.end = typeof target.selectionEnd === 'number' ? target.selectionEnd : state.start;
  }}

  function setNativeValue(textarea, value) {{
    const descriptor = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value');
    if (descriptor && descriptor.set) {{
      descriptor.set.call(textarea, value);
    }} else {{
      textarea.value = value;
    }}
  }}

  window.insertEmojiToPrompt = function(emoji) {{
    const textarea = getPromptTextarea();
    if (!textarea || !emoji) return false;

    const currentValue = textarea.value || '';
    const start = typeof state.start === 'number' ? state.start : (typeof textarea.selectionStart === 'number' ? textarea.selectionStart : currentValue.length);
    const end = typeof state.end === 'number' ? state.end : (typeof textarea.selectionEnd === 'number' ? textarea.selectionEnd : start);
    const nextValue = currentValue.slice(0, start) + emoji + currentValue.slice(end);
    const nextCursor = start + emoji.length;

    textarea.focus();
    setNativeValue(textarea, nextValue);
    textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
    textarea.dispatchEvent(new Event('change', {{ bubbles: true }}));

    requestAnimationFrame(() => {{
      const freshTextarea = getPromptTextarea();
      if (!freshTextarea) return;
      freshTextarea.focus();
      if (freshTextarea.setSelectionRange) {{
        freshTextarea.setSelectionRange(nextCursor, nextCursor);
      }}
      rememberSelection(freshTextarea);
    }});
    return false;
  }};

  function closeReferencePresetPalettes() {{
    document.querySelectorAll('.reference-preset-color-palette[data-open="true"]').forEach((palette) => {{
      palette.setAttribute('data-open', 'false');
    }});
  }}

  function syncReferencePresetPaletteSelection(widget) {{
    if (!widget) return false;
    const trigger = widget.querySelector('.reference-preset-color-trigger');
    const slotIndex = widget.getAttribute('data-slot-index');
    const hiddenRoot = slotIndex ? document.getElementById(`{REFERENCE_PRESET_COLOR_INPUT_ID_PREFIX}${{slotIndex}}`) : null;
    const hiddenInput = hiddenRoot ? hiddenRoot.querySelector('input, textarea') : null;
    const currentColor =
      (hiddenInput && hiddenInput.value) ||
      (trigger && trigger.style && trigger.style.background) ||
      '{DEFAULT_REFERENCE_PRESET_COLOR}';
    widget.querySelectorAll('.reference-preset-color-swatch').forEach((swatch) => {{
      const isSelected = (swatch.getAttribute('data-color') || '').toLowerCase() === String(currentColor).trim().toLowerCase();
      swatch.setAttribute('data-selected', isSelected ? 'true' : 'false');
    }});
    return true;
  }}

  function syncReferencePresetColor(target) {{
    const swatch = target && target.closest ? target.closest('.reference-preset-color-swatch') : null;
    if (!swatch) return false;
    const widget = swatch.closest('.reference-preset-color-widget');
    if (!widget) return false;
    const slotIndex = widget.getAttribute('data-slot-index');
    if (!slotIndex) return false;
    const hiddenRoot = document.getElementById(`{REFERENCE_PRESET_COLOR_INPUT_ID_PREFIX}${{slotIndex}}`);
    const hiddenInput = hiddenRoot ? hiddenRoot.querySelector('input, textarea') : null;
    if (!hiddenInput) return false;
    const nextValue = swatch.getAttribute('data-color') || '{DEFAULT_REFERENCE_PRESET_COLOR}';
    if (hiddenInput.value !== nextValue) {{
      hiddenInput.value = nextValue;
      hiddenInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
      hiddenInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
    }}
    syncReferencePresetPaletteSelection(widget);
    closeReferencePresetPalettes();
    return true;
  }}

  function getReferencePresetReorderInput() {{
    const root = document.getElementById('{REFERENCE_PRESET_REORDER_INPUT_ID}');
    return root ? root.querySelector('input, textarea') : null;
  }}

  function getPresetMetadataDropInput() {{
    const root = document.getElementById('{PRESET_METADATA_DROP_INPUT_ID}');
    return root ? root.querySelector('input, textarea') : null;
  }}

  function getReferencePresetPath(slotIndex) {{
    const root = document.getElementById('{REFERENCE_PRESET_PATH_INPUT_ID_PREFIX}' + String(slotIndex));
    const input = root ? root.querySelector('input, textarea') : null;
    return input ? String(input.value || '').trim() : '';
  }}

  function dispatchReferencePresetReorder(sourceIndex, targetIndex) {{
    if (sourceIndex === null || targetIndex === null || sourceIndex === '' || targetIndex === '') return false;
    if (String(sourceIndex) === String(targetIndex)) return false;
    const reorderInput = getReferencePresetReorderInput();
    if (!reorderInput) return false;
    const nextValue = `${{sourceIndex}}:${{targetIndex}}:${{Date.now()}}`;
    reorderInput.value = nextValue;
    reorderInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
    reorderInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
    return true;
  }}

  function dispatchPresetMetadataDrop(wavPath) {{
    const dropInput = getPresetMetadataDropInput();
    if (!dropInput) return false;
    const nextValue = String(wavPath || '').trim();
    if (!nextValue) return false;
    dropInput.value = nextValue;
    dropInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
    dropInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
    return true;
  }}

  const referencePresetDragState = window.__referencePresetDragState || {{
    pointerId: null,
    slot: null,
    slotIndex: '',
    presetPath: '',
    startX: 0,
    startY: 0,
    lastX: 0,
    lastY: 0,
    active: false,
  }};
  window.__referencePresetDragState = referencePresetDragState;

  function ensureReferencePresetDragPreview() {{
    let preview = document.getElementById('reference-preset-drag-preview');
    if (!preview) {{
      preview = document.createElement('div');
      preview.id = 'reference-preset-drag-preview';
      preview.className = 'reference-preset-drag-preview';
      preview.style.display = 'none';
      preview.innerHTML = '<span class="preview-label"></span><span class="preview-sub">Drag preset</span>';
      document.body.appendChild(preview);
    }}
    return preview;
  }}

  function updateReferencePresetDragPreview(clientX, clientY) {{
    const preview = ensureReferencePresetDragPreview();
    const label = preview.querySelector('.preview-label');
    const button = referencePresetDragState.slot
      ? referencePresetDragState.slot.querySelector('.{REFERENCE_PRESET_SET_BUTTON_CLASS}')
      : null;
    if (label) {{
      label.textContent = button
        ? String(button.textContent || '').trim()
        : ('Preset ' + String(referencePresetDragState.slotIndex || '').trim());
    }}
    preview.style.display = 'block';
    preview.style.transform = 'translate(' + Math.round(clientX + 14) + 'px, ' + Math.round(clientY + 12) + 'px)';
  }}

  function hideReferencePresetDragPreview() {{
    const preview = document.getElementById('reference-preset-drag-preview');
    if (preview) {{
      preview.style.display = 'none';
    }}
  }}

  function clearReferencePresetDragUi() {{
    document.querySelectorAll('.reference-preset-slot[data-drop-target="true"]').forEach((targetSlot) => {{
      targetSlot.setAttribute('data-drop-target', 'false');
    }});
    document.querySelectorAll('.reference-preset-slot[data-dragging="true"]').forEach((draggingSlot) => {{
      draggingSlot.setAttribute('data-dragging', 'false');
      draggingSlot.setAttribute('data-drag-path', '');
    }});
    const promptRoot = document.getElementById(promptBoxId);
    if (promptRoot) {{
      promptRoot.setAttribute('data-preset-drop-target', 'false');
    }}
    document.body.style.userSelect = '';
    hideReferencePresetDragPreview();
  }}

  function resetReferencePresetDragState() {{
    clearReferencePresetDragUi();
    referencePresetDragState.pointerId = null;
    referencePresetDragState.slot = null;
    referencePresetDragState.slotIndex = '';
    referencePresetDragState.presetPath = '';
    referencePresetDragState.startX = 0;
    referencePresetDragState.startY = 0;
    referencePresetDragState.lastX = 0;
    referencePresetDragState.lastY = 0;
    referencePresetDragState.active = false;
  }}

  function updateReferencePresetDropTarget(clientX, clientY) {{
    clearReferencePresetDragUi();
    if (!referencePresetDragState.slot) return {{ targetSlot: null, insidePrompt: false }};

    referencePresetDragState.slot.setAttribute('data-dragging', 'true');
    referencePresetDragState.slot.setAttribute('data-drag-path', referencePresetDragState.presetPath || '');

    const hovered = document.elementFromPoint(clientX, clientY);
    const targetSlot = hovered && hovered.closest ? hovered.closest('.reference-preset-slot') : null;
    const promptRoot = document.getElementById(promptBoxId);
    const insidePrompt = !!(promptRoot && hovered && hovered.closest && hovered.closest(`#${{promptBoxId}}`));

    if (targetSlot && targetSlot !== referencePresetDragState.slot) {{
      targetSlot.setAttribute('data-drop-target', 'true');
    }} else if (insidePrompt && referencePresetDragState.presetPath) {{
      promptRoot.setAttribute('data-preset-drop-target', 'true');
    }}

    return {{ targetSlot, insidePrompt }};
  }}

  function ensureReferencePresetPointerBindings() {{
    if (window.__referencePresetPointerBindingsBound === true) return;
    window.__referencePresetPointerBindingsBound = true;

    window.addEventListener('pointermove', (event) => {{
      if (referencePresetDragState.pointerId === null || event.pointerId !== referencePresetDragState.pointerId) return;

      referencePresetDragState.lastX = event.clientX;
      referencePresetDragState.lastY = event.clientY;

      if (!referencePresetDragState.active) {{
        const dx = Math.abs(event.clientX - referencePresetDragState.startX);
        const dy = Math.abs(event.clientY - referencePresetDragState.startY);
        if (Math.max(dx, dy) < 6) return;
        referencePresetDragState.active = true;
        document.body.style.userSelect = 'none';
      }}

      updateReferencePresetDropTarget(event.clientX, event.clientY);
      updateReferencePresetDragPreview(event.clientX, event.clientY);
    }}, true);

    window.addEventListener('pointerup', (event) => {{
      if (referencePresetDragState.pointerId === null || event.pointerId !== referencePresetDragState.pointerId) return;
      const sourceIndex = referencePresetDragState.slotIndex;
      const presetPath = referencePresetDragState.presetPath;
      const wasActive = referencePresetDragState.active;
      if (!wasActive) {{
        resetReferencePresetDragState();
        return;
      }}
      const {{ targetSlot, insidePrompt }} = updateReferencePresetDropTarget(event.clientX, event.clientY);
      resetReferencePresetDragState();
      if (insidePrompt && presetPath) {{
        dispatchPresetMetadataDrop(presetPath);
        return;
      }}
      const targetIndex = targetSlot ? targetSlot.getAttribute('data-slot-index') : '';
      dispatchReferencePresetReorder(sourceIndex, targetIndex);
    }}, true);

    window.addEventListener('pointercancel', () => {{
      resetReferencePresetDragState();
    }}, true);

    window.addEventListener('blur', () => {{
      resetReferencePresetDragState();
    }});
  }}

  function bindReferencePresetDragAndDrop() {{
    ensureReferencePresetPointerBindings();
    document.querySelectorAll('.reference-preset-slot').forEach((slot) => {{
      if (!slot.getAttribute('data-slot-index') && slot.id && slot.id.startsWith('{REFERENCE_PRESET_SLOT_ID_PREFIX}')) {{
        slot.setAttribute('data-slot-index', slot.id.slice('{REFERENCE_PRESET_SLOT_ID_PREFIX}'.length));
      }}
      if (slot.dataset.dragBound === 'true') return;
      slot.dataset.dragBound = 'true';
      slot.removeAttribute('draggable');

      const dragHandle = slot.querySelector('.{REFERENCE_PRESET_SET_BUTTON_CLASS}');
      const startDrag = (event) => {{
        if (event.button !== 0) return;
        if (!dragHandle || !dragHandle.contains(event.target)) return;
        if (referencePresetDragState.pointerId !== null) return;
        const slotIndex = slot.getAttribute('data-slot-index');
        const presetPath = getReferencePresetPath(slotIndex);
        if (!slotIndex || !presetPath) return;
        referencePresetDragState.pointerId = event.pointerId;
        referencePresetDragState.slot = slot;
        referencePresetDragState.slotIndex = slotIndex;
        referencePresetDragState.presetPath = presetPath;
        referencePresetDragState.startX = event.clientX;
        referencePresetDragState.startY = event.clientY;
        referencePresetDragState.lastX = event.clientX;
        referencePresetDragState.lastY = event.clientY;
        referencePresetDragState.active = false;
      }};

      dragHandle?.addEventListener('pointerdown', startDrag, true);
    }});
  }}

  function bindPromptPresetDrop() {{
    const promptRoot = document.getElementById(promptBoxId);
    if (!promptRoot || promptRoot.dataset.presetDropBound === 'true') return;
    promptRoot.dataset.presetDropBound = 'true';
  }}

  function syncReferencePresetInlinePreviews() {{
    document.querySelectorAll('.reference-preset-slot').forEach((slot) => {{
      const source = slot.querySelector('.reference-preset-prompt-preview');
      const uploadRoot = slot.querySelector('.{REFERENCE_PRESET_UPLOAD_CLASS}');
      if (!source || !uploadRoot) return;

      let inlinePreview = uploadRoot.querySelector('.reference-preset-upload-inline-preview');
      if (!inlinePreview) {{
        inlinePreview = document.createElement('div');
        inlinePreview.className = 'reference-preset-upload-inline-preview is-empty';
        uploadRoot.appendChild(inlinePreview);
      }}

      const previewText =
        source.getAttribute('data-preview-text') ||
        source.textContent ||
        '';

      inlinePreview.textContent = previewText;
      inlinePreview.classList.toggle('is-empty', !String(previewText).trim());
    }});
  }}

  document.addEventListener('click', (event) => {{
    const uploadRoot = event.target && event.target.closest ? event.target.closest('.{REFERENCE_PRESET_UPLOAD_CLASS}') : null;
    if (uploadRoot) {{
      const target = event.target;
      const previewClearButton = target && target.closest ? target.closest('[aria-label*="Clear"], [aria-label*="Remove"], [data-testid*="remove"], [data-testid*="clear"], .delete-button, .remove-button') : null;
      if (!previewClearButton) {{
        event.preventDefault();
        event.stopPropagation();
        return;
      }}
    }}
    const emojiBtn = event.target && event.target.closest ? event.target.closest('.emoji-btn[data-emoji]') : null;
    if (emojiBtn) {{
      event.preventDefault();
      event.stopPropagation();
      window.insertEmojiToPrompt(emojiBtn.dataset.emoji);
      return;
    }}
    const swatchBtn = event.target && event.target.closest ? event.target.closest('.reference-preset-color-swatch') : null;
    if (swatchBtn) {{
      event.preventDefault();
      event.stopPropagation();
      syncReferencePresetColor(swatchBtn);
      return;
    }}
    const colorTrigger = event.target && event.target.closest ? event.target.closest('.reference-preset-color-trigger') : null;
    if (colorTrigger) {{
      event.preventDefault();
      event.stopPropagation();
      if (colorTrigger.hasAttribute('disabled')) return;
      const widget = colorTrigger.closest('.reference-preset-color-widget');
      const palette = widget ? widget.querySelector('.reference-preset-color-palette') : null;
      if (!palette) return;
      const willOpen = palette.getAttribute('data-open') !== 'true';
      closeReferencePresetPalettes();
      if (willOpen) {{
        syncReferencePresetPaletteSelection(widget);
      }}
      palette.setAttribute('data-open', willOpen ? 'true' : 'false');
      return;
    }}
    closeReferencePresetPalettes();
    if (event.target && event.target.closest && event.target.closest(promptSelector)) {{
      rememberSelection(event.target);
    }}
  }}, true);

  document.addEventListener('focusin', (event) => {{
    if (event.target && event.target.closest && event.target.closest(promptSelector)) {{
      rememberSelection(event.target);
    }}
  }}, true);

  document.addEventListener('keyup', (event) => {{
    if (event.target && event.target.closest && event.target.closest(promptSelector)) {{
      rememberSelection(event.target);
    }}
  }}, true);

  document.addEventListener('input', (event) => {{
    if (event.target && event.target.closest && event.target.closest(promptSelector)) {{
      rememberSelection(event.target);
    }}
  }}, true);

  document.addEventListener('select', (event) => {{
    if (event.target && event.target.closest && event.target.closest(promptSelector)) {{
      rememberSelection(event.target);
    }}
  }}, true);

  function bindPresetInteractions() {{
    bindReferencePresetDragAndDrop();
    bindPromptPresetDrop();
    syncReferencePresetInlinePreviews();
  }}

  document.addEventListener('DOMContentLoaded', bindPresetInteractions);
  document.addEventListener('gradio:render', bindPresetInteractions);
  setInterval(bindPresetInteractions, 1200);
}})();
</script>
"""




def _settings_path() -> Path:
    return APP_DATA_ROOT / SETTINGS_FILE_NAME


def _default_reference_presets() -> list[dict[str, str]]:
    return [
        {
            "path": "",
            "name": "",
            "color": DEFAULT_REFERENCE_PRESET_COLOR,
            "prompt_preview": "",
        }
        for _ in range(REFERENCE_PRESET_COUNT)
    ]


def _normalize_reference_preset_color(raw_color: str | None) -> str:
    text = str(raw_color or "").strip()
    if not text:
        return DEFAULT_REFERENCE_PRESET_COLOR
    if not text.startswith("#"):
        text = f"#{text}"
    if len(text) == 4 and all(ch in "0123456789abcdefABCDEF" for ch in text[1:]):
        text = "#" + "".join(ch * 2 for ch in text[1:])
    if len(text) != 7 or any(ch not in "0123456789abcdefABCDEF" for ch in text[1:]):
        return DEFAULT_REFERENCE_PRESET_COLOR
    return text.lower()


def _normalize_reference_presets(raw_value: object) -> list[dict[str, str]]:
    defaults = _default_reference_presets()
    if not isinstance(raw_value, list):
        return defaults

    normalized = defaults.copy()
    for index, item in enumerate(raw_value[:REFERENCE_PRESET_COUNT]):
        if not isinstance(item, dict):
            continue
        normalized[index] = {
            "path": str(item.get("path", "") or "").strip(),
            "name": str(item.get("name", "") or "").strip(),
            "color": _normalize_reference_preset_color(item.get("color")),
            "prompt_preview": str(item.get("prompt_preview", "") or "").strip(),
        }
    return normalized


def _normalize_active_reference_preset_slot(raw_value: object) -> int | None:
    if raw_value is None or str(raw_value).strip() == "":
        return None
    try:
        slot_index = int(raw_value)
    except (TypeError, ValueError):
        return None
    if 0 <= slot_index < REFERENCE_PRESET_COUNT:
        return slot_index
    return None


def _load_app_settings() -> dict[str, object]:
    defaults: dict[str, object] = {
        "output_dir": DEFAULT_OUTPUT_DIR,
        "save_dir": "",
        "reference_presets": _default_reference_presets(),
        "active_reference_preset_slot": None,
    }

    path = _settings_path()
    if not path.exists():
        return defaults

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[gradio] failed to read settings: {e}", flush=True)
        return defaults

    if not isinstance(data, dict):
        return defaults

    return {
        "output_dir": str(data.get("output_dir", defaults["output_dir"]) or defaults["output_dir"]),
        "save_dir": str(data.get("save_dir", defaults["save_dir"]) or ""),
        "reference_presets": _normalize_reference_presets(data.get("reference_presets")),
        "active_reference_preset_slot": _normalize_active_reference_preset_slot(
            data.get("active_reference_preset_slot")
        ),
    }


def _save_app_settings(
    output_dir: str | None = None,
    save_dir: str | None = None,
    reference_presets: list[dict[str, str]] | None = None,
    active_reference_preset_slot: int | None | object = ...,
) -> None:
    current = _load_app_settings()

    payload = {
        "output_dir": str(
            current.get("output_dir", DEFAULT_OUTPUT_DIR) if output_dir is None else output_dir
        ).strip()
        or DEFAULT_OUTPUT_DIR,
        "save_dir": str(current.get("save_dir", "") if save_dir is None else save_dir).strip(),
        "reference_presets": _normalize_reference_presets(
            current.get("reference_presets") if reference_presets is None else reference_presets
        ),
        "active_reference_preset_slot": _normalize_active_reference_preset_slot(
            current.get("active_reference_preset_slot")
            if active_reference_preset_slot is ...
            else active_reference_preset_slot
        ),
    }

    path = _settings_path()
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[gradio] failed to write settings: {e}", flush=True)


def _persist_directory_settings(output_dir: str | None, save_dir: str | None) -> None:
    normalized_output_dir = str(
        _resolve_internal_output_dir(output_dir, default_dir=DEFAULT_OUTPUT_DIR).relative_to(
            APP_DATA_ROOT
        )
    )
    _save_app_settings(output_dir=normalized_output_dir, save_dir=save_dir)


def _reference_preset_dir() -> Path:
    return APP_DATA_ROOT / REFERENCE_PRESET_DIR_NAME


def _truncate_reference_preset_name(name: str, max_chars: int = REFERENCE_PRESET_LABEL_MAX_CHARS) -> str:
    text = str(name or "").strip()
    if text == "":
        return "Empty"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _reference_preset_button_label(preset: dict[str, str], slot_index: int) -> str:
    display_name = str(preset.get("name", "")).strip()
    if display_name == "":
        display_name = f"Preset {slot_index + 1}"
    else:
        display_name = Path(display_name).stem
    return _truncate_reference_preset_name(display_name)


def _reference_preset_has_file(preset: dict[str, str]) -> bool:
    path_text = str(preset.get("path", "") or "").strip()
    return path_text != "" and Path(path_text).exists()


def _reference_preset_button_update(preset: dict[str, str], slot_index: int) -> gr.Button:
    return gr.update(
        value=_reference_preset_button_label(preset, slot_index),
        interactive=_reference_preset_has_file(preset),
    )


def _reference_preset_clear_button_update(preset: dict[str, str]) -> gr.Button:
    return gr.update(interactive=_reference_preset_has_file(preset))


def _reference_preset_notice_update(message: str = "") -> gr.HTML:
    text = str(message or "").strip()
    if text == "":
        return gr.update(value="")
    return gr.update(value=f'<div class="preset-notice">{html.escape(text)}</div>')


def _reference_preset_color_picker_html(preset: dict[str, str], slot_index: int) -> str:
    color = _normalize_reference_preset_color(preset.get("color"))
    disabled_attr = " disabled" if not _reference_preset_has_file(preset) else ""
    swatches = "".join(
        (
            f'<button type="button" class="reference-preset-color-swatch" '
            f'data-color="{palette_color}" data-selected="{"true" if palette_color == color else "false"}" '
            f'style="background:{palette_color}" aria-label="{palette_color}"></button>'
        )
        for palette_color in REFERENCE_PRESET_PALETTE_COLORS
    )
    return "".join(
        [
            f'<div class="reference-preset-color-widget" data-slot-index="{slot_index}">',
            (
                f'<button type="button" id="{REFERENCE_PRESET_COLOR_BUTTON_ID_PREFIX}{slot_index}" '
                f'class="{REFERENCE_PRESET_COLOR_BUTTON_CLASS} reference-preset-color-trigger" '
                f'style="background:{html.escape(color, quote=True)}"{disabled_attr}></button>'
            ),
            f'<div class="reference-preset-color-palette" data-open="false">{swatches}</div>',
            "</div>",
        ]
    )


def _reference_preset_color_button_update(preset: dict[str, str], slot_index: int) -> gr.HTML:
    return gr.update(value=_reference_preset_color_picker_html(preset, slot_index))


def _reference_preset_prompt_preview_text(prompt_text: str | None) -> str:
    text = " ".join(str(prompt_text or "").strip().split())
    return text[:72]


def _reference_preset_prompt_preview_html(preset: dict[str, str]) -> str:
    prompt_preview = _reference_preset_prompt_preview_text(preset.get("prompt_preview"))
    classes = "reference-preset-prompt-preview"
    if prompt_preview == "":
        classes += " is-empty"
    return (
        f'<div class="{classes}" data-preview-text="{html.escape(prompt_preview, quote=True)}">'
        f"{html.escape(prompt_preview)}</div>"
    )


def _reference_preset_prompt_preview_update(preset: dict[str, str]) -> gr.HTML:
    return gr.update(value=_reference_preset_prompt_preview_html(preset))


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    text = _normalize_reference_preset_color(hex_color).lstrip("#")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


def _adjust_hex_color(hex_color: str, amount: float) -> str:
    red, green, blue = _hex_to_rgb(hex_color)

    def adjust(channel: int) -> int:
        if amount >= 0:
            return max(0, min(255, int(channel + (255 - channel) * amount)))
        return max(0, min(255, int(channel * (1.0 + amount))))

    adjusted = (adjust(red), adjust(green), adjust(blue))
    return "#{:02x}{:02x}{:02x}".format(*adjusted)


def _reference_preset_text_color(hex_color: str) -> str:
    red, green, blue = _hex_to_rgb(hex_color)
    luminance = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    return "#111111" if luminance >= 186 else "#ffffff"


def _build_reference_preset_style_html(
    reference_presets: object,
    selected_slot_index: int | None = None,
) -> str:
    presets = _normalize_reference_presets(reference_presets)
    lines = ["<style>"]
    for index, preset in enumerate(presets):
        has_file = _reference_preset_has_file(preset)
        bg = _normalize_reference_preset_color(preset.get("color")) if has_file else "#ffffff"
        hover = _adjust_hex_color(bg, -0.08)
        active = _adjust_hex_color(bg, -0.18)
        border = _adjust_hex_color(bg, -0.14)
        selected_bg = _adjust_hex_color(bg, -0.26)
        selected_border = _adjust_hex_color(bg, -0.36)
        text_color = _reference_preset_text_color(bg)
        button_id = f"{REFERENCE_PRESET_BUTTON_ID_PREFIX}{index}"
        lines.extend(
            [
                f"#{button_id} {{ background: {bg} !important; border: 1px solid {border} !important; color: {text_color} !important; }}",
                f"#{button_id}:hover {{ background: {hover} !important; }}",
                f"#{button_id}:active {{ background: {active} !important; }}",
                f"#{button_id}[disabled] {{ background: {hover} !important; border-color: {border} !important; color: {text_color} !important; pointer-events: none !important; cursor: default !important; }}",
                f"#{REFERENCE_PRESET_COLOR_BUTTON_ID_PREFIX}{index} {{ background: {bg} !important; border: 1px solid {border} !important; color: {text_color} !important; }}",
                f"#{REFERENCE_PRESET_COLOR_BUTTON_ID_PREFIX}{index}:hover {{ background: {hover} !important; }}",
                f"#{REFERENCE_PRESET_COLOR_BUTTON_ID_PREFIX}{index}:active {{ background: {active} !important; }}",
                f"#{REFERENCE_PRESET_COLOR_BUTTON_ID_PREFIX}{index}[disabled] {{ background: {hover} !important; border-color: {border} !important; color: {text_color} !important; }}",
            ]
        )
        if selected_slot_index == index:
            lines.extend(
                [
                    f"#{button_id} {{ background: {selected_bg} !important; border: 1px solid {selected_border} !important; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.34) inset, 0 0 0 2px rgba(255, 255, 255, 0.14) !important; transform: translateY(2px); filter: saturate(0.96) brightness(0.97) !important; }}",
                    f"#{button_id}:hover {{ background: {selected_bg} !important; }}",
                ]
            )
    lines.append("</style>")
    return "\n".join(lines)


def _find_matching_reference_preset_slot(
    uploaded_audio: str | None,
    current_presets: object,
) -> int | None:
    if uploaded_audio is None or str(uploaded_audio).strip() == "":
        return None

    uploaded_name = Path(str(uploaded_audio)).name.strip().lower()
    uploaded_path = None
    try:
        uploaded_path = str(Path(str(uploaded_audio)).resolve())
    except Exception:
        uploaded_path = None

    for index, preset in enumerate(_normalize_reference_presets(current_presets)):
        path_text = str(preset.get("path", "") or "").strip()
        if path_text == "":
            continue
        preset_name = Path(path_text).name.strip().lower()
        if uploaded_name != "" and preset_name == uploaded_name:
            return index
        try:
            if uploaded_path is not None and str(Path(path_text).resolve()) == uploaded_path:
                return index
        except Exception:
            continue
    return None


def _handle_reference_audio_change(
    uploaded_audio: str | None,
    current_presets: object,
    preserve_selection: bool,
) -> tuple[str, int | None, str, bool, gr.HTML]:
    presets = _normalize_reference_presets(current_presets)
    selected_slot_index = _find_matching_reference_preset_slot(uploaded_audio, presets)
    _save_app_settings(active_reference_preset_slot=selected_slot_index)
    return (
        _display_ref_audio_name(uploaded_audio),
        selected_slot_index,
        _build_reference_preset_style_html(presets, selected_slot_index),
        False,
        _reference_preset_notice_update(),
    )


def _reference_preset_storage_path(source_name: str) -> Path:
    file_name = Path(str(source_name or "")).name.strip()
    if file_name == "":
        file_name = "reference.wav"
    elif Path(file_name).suffix == "":
        file_name += ".wav"
    return _reference_preset_dir() / file_name


def _set_reference_preset_color(
    selected_color: str,
    output_dir: str,
    save_dir: str,
    current_presets: object,
    selected_slot_index: int | None,
    slot_index: int,
) -> tuple[gr.Button, list[dict[str, str]], str, int | None, gr.HTML]:
    presets = _normalize_reference_presets(current_presets)
    preset = presets[slot_index]
    next_color = _normalize_reference_preset_color(selected_color)
    presets[slot_index] = {
        "path": str(preset.get("path", "") or "").strip(),
        "name": str(preset.get("name", "") or "").strip(),
        "color": next_color,
    }
    _save_app_settings(
        output_dir=output_dir,
        save_dir=save_dir,
        reference_presets=presets,
        active_reference_preset_slot=selected_slot_index,
    )
    return (
        _reference_preset_button_update(presets[slot_index], slot_index),
        presets,
        _build_reference_preset_style_html(presets, selected_slot_index),
        selected_slot_index,
        _reference_preset_notice_update(),
    )


def _swap_reference_presets(
    reorder_value: str,
    output_dir: str,
    save_dir: str,
    current_presets: object,
    selected_slot_index: int | None,
) -> tuple[object, ...]:
    text = str(reorder_value or "").strip()
    presets = _normalize_reference_presets(current_presets)
    if text == "":
        return (
            *[_reference_preset_prompt_preview_update(preset) for preset in presets],
            *[gr.update(value=str(preset.get("path", "") or "").strip()) for preset in presets],
            *[gr.update(value=_normalize_reference_preset_color(preset.get("color"))) for preset in presets],
            *[_reference_preset_button_update(preset, index) for index, preset in enumerate(presets)],
            *[_reference_preset_clear_button_update(preset) for preset in presets],
            *[_reference_preset_color_button_update(preset, index) for index, preset in enumerate(presets)],
            presets,
            selected_slot_index,
            _build_reference_preset_style_html(presets, selected_slot_index),
            gr.update(value=None if selected_slot_index is None else str(presets[selected_slot_index].get("path", "") or "").strip() or None),
            _display_ref_audio_name(None if selected_slot_index is None else str(presets[selected_slot_index].get("path", "") or "").strip() or None),
            _reference_preset_notice_update(),
            gr.update(value=""),
        )

    parts = text.split(":")
    if len(parts) < 2:
        return (
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            presets,
            selected_slot_index,
            _build_reference_preset_style_html(presets, selected_slot_index),
            gr.skip(),
            gr.skip(),
            _reference_preset_notice_update(),
            gr.update(value=""),
        )

    try:
        source_index = int(parts[0])
        target_index = int(parts[1])
    except ValueError:
        return (
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            presets,
            selected_slot_index,
            _build_reference_preset_style_html(presets, selected_slot_index),
            gr.skip(),
            gr.skip(),
            _reference_preset_notice_update("Preset reorder request was invalid."),
            gr.update(value=""),
        )

    if not (0 <= source_index < REFERENCE_PRESET_COUNT and 0 <= target_index < REFERENCE_PRESET_COUNT):
        return (
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            *[gr.skip() for _ in range(REFERENCE_PRESET_COUNT)],
            presets,
            selected_slot_index,
            _build_reference_preset_style_html(presets, selected_slot_index),
            gr.skip(),
            gr.skip(),
            _reference_preset_notice_update("Preset reorder request was out of range."),
            gr.update(value=""),
        )

    if source_index != target_index:
        presets[source_index], presets[target_index] = presets[target_index], presets[source_index]
        if selected_slot_index == source_index:
            selected_slot_index = target_index
        elif selected_slot_index == target_index:
            selected_slot_index = source_index

    _save_app_settings(
        output_dir=output_dir,
        save_dir=save_dir,
        reference_presets=presets,
        active_reference_preset_slot=selected_slot_index,
    )

    uploaded_audio = None
    ref_audio_name = ""
    if selected_slot_index is not None:
        active_path = str(presets[selected_slot_index].get("path", "") or "").strip()
        if active_path != "" and Path(active_path).exists():
            uploaded_audio = active_path
            ref_audio_name = Path(active_path).name
        else:
            selected_slot_index = None

    return (
        *[_reference_preset_prompt_preview_update(preset) for preset in presets],
        *[gr.update(value=str(preset.get("path", "") or "").strip()) for preset in presets],
        *[gr.update(value=_normalize_reference_preset_color(preset.get("color"))) for preset in presets],
        *[_reference_preset_button_update(preset, index) for index, preset in enumerate(presets)],
        *[_reference_preset_clear_button_update(preset) for preset in presets],
        *[_reference_preset_color_button_update(preset, index) for index, preset in enumerate(presets)],
        presets,
        selected_slot_index,
        _build_reference_preset_style_html(presets, selected_slot_index),
        gr.update(value=uploaded_audio),
        ref_audio_name,
        _reference_preset_notice_update(),
        gr.update(value=""),
    )


def _register_reference_preset(
    uploaded_file_path: str | None,
    output_dir: str,
    save_dir: str,
    current_presets: object,
    selected_slot_index: int | None,
    slot_index: int,
) -> tuple[gr.File, gr.Textbox, gr.Button, gr.Button, gr.HTML, gr.HTML, list[dict[str, str]], str, int | None, gr.HTML]:
    if uploaded_file_path is None or str(uploaded_file_path).strip() == "":
        presets = _normalize_reference_presets(current_presets)
        return (
            gr.update(value=None),
            gr.update(value=str(presets[slot_index].get("path", "") or "").strip()),
            _reference_preset_button_update(presets[slot_index], slot_index),
            _reference_preset_clear_button_update(presets[slot_index]),
            _reference_preset_color_button_update(presets[slot_index], slot_index),
            _reference_preset_prompt_preview_update(presets[slot_index]),
            presets,
            _build_reference_preset_style_html(presets, selected_slot_index),
            selected_slot_index,
            gr.skip(),
        )

    src = Path(str(uploaded_file_path))
    if not src.exists():
        raise ValueError("Dropped WAV file was not found.")

    presets = _normalize_reference_presets(current_presets)
    duplicate_slot = next(
        (
            index
            for index, preset in enumerate(presets)
            if index != slot_index and str(preset.get("name", "") or "").strip().lower() == src.name.lower()
        ),
        None,
    )
    if duplicate_slot is not None:
        warning_message = (
            f"'{src.name}' is already registered in Preset {duplicate_slot + 1}. "
            "The same file name cannot be registered to multiple preset buttons."
        )
        print(f"[gradio] {warning_message}", flush=True)
        gr.Warning(warning_message)
        return (
            gr.update(value=None),
            gr.update(value=str(presets[slot_index].get("path", "") or "").strip()),
            _reference_preset_button_update(presets[slot_index], slot_index),
            _reference_preset_clear_button_update(presets[slot_index]),
            _reference_preset_color_button_update(presets[slot_index], slot_index),
            _reference_preset_prompt_preview_update(presets[slot_index]),
            presets,
            _build_reference_preset_style_html(presets, selected_slot_index),
            selected_slot_index,
            _reference_preset_notice_update(warning_message),
        )

    dst = _reference_preset_storage_path(src.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    presets[slot_index] = {
        "path": str(dst),
        "name": src.name,
        "color": (
            DEFAULT_REFERENCE_PRESET_ASSIGNED_COLOR
            if not _reference_preset_has_file(presets[slot_index])
            else presets[slot_index]["color"]
        ),
        "prompt_preview": _reference_preset_prompt_preview_text(
            (_read_embedded_metadata(dst) or {}).get("prompt", "")
        ),
    }

    _save_app_settings(
        output_dir=output_dir,
        save_dir=save_dir,
        reference_presets=presets,
        active_reference_preset_slot=selected_slot_index,
    )
    return (
        gr.update(value=None),
        gr.update(value=str(presets[slot_index].get("path", "") or "").strip()),
        _reference_preset_button_update(presets[slot_index], slot_index),
        _reference_preset_clear_button_update(presets[slot_index]),
        _reference_preset_color_button_update(presets[slot_index], slot_index),
        _reference_preset_prompt_preview_update(presets[slot_index]),
        presets,
        _build_reference_preset_style_html(presets, selected_slot_index),
        selected_slot_index,
        _reference_preset_notice_update(),
    )


def _apply_reference_preset(
    current_presets: object,
    uploaded_audio: str | None,
    selected_slot_index: int | None,
    slot_index: int,
) -> tuple[gr.Audio, str, int | None, str, bool, gr.HTML]:
    presets = _normalize_reference_presets(current_presets)
    preset = presets[slot_index]
    path_text = str(preset.get("path", "") or "").strip()
    if path_text == "":
        raise ValueError(f"Reference preset {slot_index + 1} is empty.")

    wav_path = Path(path_text)
    if not wav_path.exists():
        raise ValueError(f"Reference preset file was not found: {wav_path}")

    if selected_slot_index == slot_index:
        _save_app_settings(active_reference_preset_slot=None)
        return (
            gr.update(value=None),
            "",
            None,
            _build_reference_preset_style_html(presets, None),
            False,
            _reference_preset_notice_update(),
        )

    _save_app_settings(active_reference_preset_slot=slot_index)
    return (
        gr.update(value=str(wav_path)),
        wav_path.name,
        slot_index,
        _build_reference_preset_style_html(presets, slot_index),
        True,
        _reference_preset_notice_update(),
    )


def _same_resolved_path(path_a: str | None, path_b: str | None) -> bool:
    if str(path_a or "").strip() == "" or str(path_b or "").strip() == "":
        return False
    try:
        return Path(str(path_a)).resolve() == Path(str(path_b)).resolve()
    except Exception:
        return False


def _clear_reference_preset(
    uploaded_audio: str | None,
    output_dir: str,
    save_dir: str,
    current_presets: object,
    selected_slot_index: int | None,
    slot_index: int,
) -> tuple[gr.Textbox, gr.Button, gr.Button, gr.HTML, gr.HTML, list[dict[str, str]], str, int | None, gr.Audio, str, gr.HTML]:
    presets = _normalize_reference_presets(current_presets)
    preset = presets[slot_index]
    cleared_path = str(preset.get("path", "") or "").strip()

    if cleared_path != "":
        try:
            path_obj = Path(cleared_path)
            if path_obj.exists():
                path_obj.unlink()
        except Exception:
            pass

    presets[slot_index] = {
        "path": "",
        "name": "",
        "color": preset.get("color", DEFAULT_REFERENCE_PRESET_COLOR),
        "prompt_preview": "",
    }

    next_selected_slot = None if selected_slot_index == slot_index else selected_slot_index
    next_uploaded_audio = uploaded_audio
    next_ref_audio_name = _display_ref_audio_name(uploaded_audio)

    if _same_resolved_path(uploaded_audio, cleared_path):
        next_uploaded_audio = None
        next_ref_audio_name = ""
        next_selected_slot = None

    _save_app_settings(
        output_dir=output_dir,
        save_dir=save_dir,
        reference_presets=presets,
        active_reference_preset_slot=next_selected_slot,
    )
    return (
        gr.update(value=""),
        _reference_preset_button_update(presets[slot_index], slot_index),
        _reference_preset_clear_button_update(presets[slot_index]),
        _reference_preset_color_button_update(presets[slot_index], slot_index),
        _reference_preset_prompt_preview_update(presets[slot_index]),
        presets,
        _build_reference_preset_style_html(presets, next_selected_slot),
        next_selected_slot,
        gr.update(value=next_uploaded_audio),
        next_ref_audio_name,
        _reference_preset_notice_update(),
    )


def _build_emoji_panel_html() -> str:
    sections: list[str] = []
    for title, emojis in EMOJI_GROUPS:
        buttons = "".join(
            (
                f'<button type="button" class="emoji-btn" data-emoji="{emoji}" '
                f'aria-label="{emoji}" title="{html.escape(EMOJI_TOOLTIPS.get(emoji, ""), quote=True)}">'
                f'{emoji}</button>'
            )
            for emoji in emojis
        )
        sections.append(
            "".join(
                [
                    '<div class="emoji-section">',
                    f'<div class="emoji-title">[{title}]</div>',
                    f'<div class="emoji-grid">{buttons}</div>',
                    '</div>',
                ]
            )
        )
    return f'<div id="{EMOJI_PANEL_ID}">{"".join(sections)}</div>'


def _default_checkpoint() -> str:
    candidates = sorted(
        [
            *Path(".").glob("**/checkpoint_*.pt"),
            *Path(".").glob("**/checkpoint_*.safetensors"),
        ]
    )
    if not candidates:
        return "Aratako/Irodori-TTS-500M-v2"
    return str(candidates[-1])


def _default_model_device() -> str:
    return default_runtime_device()


def _default_codec_device() -> str:
    return default_runtime_device()


def _precision_choices_for_device(device: str) -> list[str]:
    return list_available_runtime_precisions(device)


def _on_model_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])


def _on_codec_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])


def _parse_optional_float(raw: str | None, label: str) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a float or blank.") from exc


def _parse_optional_int(raw: str | None, label: str) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an int or blank.") from exc


def _format_timings(stage_timings: list[tuple[str, float]], total_to_decode: float) -> str:
    lines = [
        "[timing] ---- request ----",
        *[f"[timing] {name}: {sec * 1000.0:.1f} ms" for name, sec in stage_timings],
        f"[timing] total_to_decode: {total_to_decode:.3f} s",
    ]
    return "\n".join(lines)


def _resolve_ref_wav(uploaded_audio: str | None) -> str | None:
    if uploaded_audio is not None and str(uploaded_audio).strip() != "":
        return str(uploaded_audio)
    return None


def _resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint is required.")

    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        return checkpoint

    resolved = hf_hub_download(repo_id=checkpoint, filename="model.safetensors")
    print(f"[gradio] checkpoint: hf://{checkpoint} -> {resolved}", flush=True)
    return str(resolved)


def _extract_prompt_text(value: dict | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text", "")
        return "" if text is None else str(text)
    return str(value)


def _extract_first_file_path(value: dict | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        files = value.get("files") or []
        if not files:
            return None
        first = files[0]
        if isinstance(first, dict):
            return first.get("path") or first.get("name")
        return str(first)
    return None


def _get_txxx_text(audio: WAVE, desc: str) -> str | None:
    if audio.tags is None:
        return None
    for frame in audio.tags.getall("TXXX"):
        if getattr(frame, "desc", "").strip().upper() == desc.upper():
            values = getattr(frame, "text", None) or []
            if values:
                return str(values[0])
    return None


def _get_comm_prompt(audio: WAVE) -> str | None:
    if audio.tags is None:
        return None
    for frame in audio.tags.getall("COMM"):
        desc = getattr(frame, "desc", "").strip().lower()
        if desc == "prompt":
            values = getattr(frame, "text", None) or []
            if values:
                return str(values[0])
    return None


def _read_embedded_metadata(wav_path: str | Path) -> dict[str, str] | None:
    if not _MUTAGEN_AVAILABLE:
        print("[gradio] mutagen not found; skipped metadata read", flush=True)
        return None

    try:
        audio = WAVE(str(wav_path))
        prompt = _get_txxx_text(audio, "PROMPT")
        if prompt is None:
            prompt = _get_comm_prompt(audio)

        return {
            "prompt": "" if prompt is None else prompt,
            "num_steps": _get_txxx_text(audio, "NUM_STEPS") or "",
            "num_candidates": _get_txxx_text(audio, "NUM_CANDIDATES") or "",
            "seed": _get_txxx_text(audio, "SEED") or "",
            "cfg_guidance_mode": _get_txxx_text(audio, "CFG_GUIDANCE_MODE") or "",
            "cfg_scale_text": _get_txxx_text(audio, "CFG_SCALE_TEXT") or "",
            "cfg_scale_speaker": _get_txxx_text(audio, "CFG_SCALE_SPEAKER") or "",
        }
    except Exception as e:
        print(f"[gradio] failed to read WAV metadata: {e}", flush=True)
        return None


def _embed_wav_metadata(
    wav_path: str | Path,
    prompt_text: str,
    num_steps: int,
    num_candidates: int,
    used_seed: int | str | None,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
) -> None:
    if not _MUTAGEN_AVAILABLE:
        print("[gradio] mutagen not found; skipped WAV metadata", flush=True)
        return

    metadata_text = "" if prompt_text is None else str(prompt_text)

    try:
        audio = WAVE(str(wav_path))
        if audio.tags is None:
            audio.add_tags()

        audio.tags.delall("COMM")
        audio.tags.delall("TIT2")
        for desc in (
            "PROMPT",
            "NUM_STEPS",
            "NUM_CANDIDATES",
            "SEED",
            "CFG_GUIDANCE_MODE",
            "CFG_SCALE_TEXT",
            "CFG_SCALE_SPEAKER",
        ):
            audio.tags.delall(f"TXXX:{desc}")

        audio.tags.add(TIT2(encoding=Encoding.UTF8, text=[Path(wav_path).stem]))

        if metadata_text != "":
            audio.tags.add(
                COMM(encoding=Encoding.UTF8, lang="jpn", desc="prompt", text=[metadata_text])
            )
            audio.tags.add(TXXX(encoding=Encoding.UTF8, desc="PROMPT", text=[metadata_text]))

        audio.tags.add(TXXX(encoding=Encoding.UTF8, desc="NUM_STEPS", text=[str(int(num_steps))]))
        audio.tags.add(
            TXXX(encoding=Encoding.UTF8, desc="NUM_CANDIDATES", text=[str(int(num_candidates))])
        )
        audio.tags.add(
            TXXX(
                encoding=Encoding.UTF8,
                desc="SEED",
                text=["" if used_seed is None else str(used_seed)],
            )
        )
        audio.tags.add(
            TXXX(encoding=Encoding.UTF8, desc="CFG_GUIDANCE_MODE", text=[str(cfg_guidance_mode)])
        )
        audio.tags.add(
            TXXX(encoding=Encoding.UTF8, desc="CFG_SCALE_TEXT", text=[str(float(cfg_scale_text))])
        )
        audio.tags.add(
            TXXX(
                encoding=Encoding.UTF8,
                desc="CFG_SCALE_SPEAKER",
                text=[str(float(cfg_scale_speaker))],
            )
        )
        audio.save(v2_version=4)
    except Exception as e:
        print(f"[gradio] failed to write WAV metadata: {e}", flush=True)


def _clamp_int(value: str, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        return default
    return max(min_value, min(max_value, parsed))


def _clamp_float(value: str, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(str(value).strip())
    except Exception:
        return default
    return max(min_value, min(max_value, parsed))


def _normalize_guidance_mode(value: str, default: str) -> str:
    allowed = {"independent", "joint", "alternating"}
    value = str(value).strip()
    if value in allowed:
        return value
    return default


def _load_prompt_metadata_from_wav_path(
    wav_path: str | None,
    current_text: str,
    current_num_steps: int,
    current_num_candidates: int,
    current_seed_raw: str,
    current_cfg_guidance_mode: str,
    current_cfg_scale_text: float,
    current_cfg_scale_speaker: float,
):
    if not wav_path:
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    print(f"[gradio] prompt metadata source: path='{wav_path}'", flush=True)

    metadata = _read_embedded_metadata(wav_path)
    if not metadata:
        print("[gradio] no embedded metadata found in dropped audio", flush=True)
        return (
            {"text": current_text, "files": []},
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    new_text = metadata.get("prompt", "")
    if new_text == "":
        new_text = current_text

    num_steps = _clamp_int(metadata.get("num_steps", ""), int(current_num_steps), 1, 120)
    num_candidates = _clamp_int(
        metadata.get("num_candidates", ""),
        int(current_num_candidates),
        1,
        MAX_GRADIO_CANDIDATES,
    )
    seed_text = metadata.get("seed", "")
    if seed_text == "":
        seed_text = str(current_seed_raw or "")
    guidance_mode = _normalize_guidance_mode(
        metadata.get("cfg_guidance_mode", ""),
        str(current_cfg_guidance_mode),
    )
    cfg_scale_text = _clamp_float(
        metadata.get("cfg_scale_text", ""),
        float(current_cfg_scale_text),
        0.0,
        10.0,
    )
    cfg_scale_speaker = _clamp_float(
        metadata.get("cfg_scale_speaker", ""),
        float(current_cfg_scale_speaker),
        0.0,
        10.0,
    )

    print("[gradio] embedded metadata found in dropped audio", flush=True)

    return (
        {"text": new_text, "files": []},
        num_steps,
        num_candidates,
        seed_text,
        guidance_mode,
        cfg_scale_text,
        cfg_scale_speaker,
    )


def _load_prompt_metadata_from_box(
    prompt_box_value: dict | str | None,
    current_num_steps: int,
    current_num_candidates: int,
    current_seed_raw: str,
    current_cfg_guidance_mode: str,
    current_cfg_scale_text: float,
    current_cfg_scale_speaker: float,
):
    wav_path = _extract_first_file_path(prompt_box_value)
    current_text = _extract_prompt_text(prompt_box_value)
    return _load_prompt_metadata_from_wav_path(
        wav_path=wav_path,
        current_text=current_text,
        current_num_steps=current_num_steps,
        current_num_candidates=current_num_candidates,
        current_seed_raw=current_seed_raw,
        current_cfg_guidance_mode=current_cfg_guidance_mode,
        current_cfg_scale_text=current_cfg_scale_text,
        current_cfg_scale_speaker=current_cfg_scale_speaker,
    )


def _hydrate_reference_preset_prompt_previews(
    presets_value: object,
) -> tuple[list[dict[str, str]], bool]:
    presets = _normalize_reference_presets(presets_value)
    changed = False
    hydrated: list[dict[str, str]] = []
    for preset in presets:
        next_preset = dict(preset)
        path_text = str(next_preset.get("path", "") or "").strip()
        prompt_preview = str(next_preset.get("prompt_preview", "") or "").strip()
        if prompt_preview == "" and path_text != "" and Path(path_text).exists():
            metadata = _read_embedded_metadata(path_text) or {}
            next_preset["prompt_preview"] = _reference_preset_prompt_preview_text(
                metadata.get("prompt", "")
            )
            if next_preset["prompt_preview"] != "":
                changed = True
        hydrated.append(next_preset)
    return hydrated, changed


def _load_prompt_metadata_from_preset_path(
    dropped_wav_path: str | None,
    current_text_value: dict | str | None,
    current_num_steps: int,
    current_num_candidates: int,
    current_seed_raw: str,
    current_cfg_guidance_mode: str,
    current_cfg_scale_text: float,
    current_cfg_scale_speaker: float,
):
    current_text = _extract_prompt_text(current_text_value)
    result = _load_prompt_metadata_from_wav_path(
        wav_path=dropped_wav_path,
        current_text=current_text,
        current_num_steps=current_num_steps,
        current_num_candidates=current_num_candidates,
        current_seed_raw=current_seed_raw,
        current_cfg_guidance_mode=current_cfg_guidance_mode,
        current_cfg_scale_text=current_cfg_scale_text,
        current_cfg_scale_speaker=current_cfg_scale_speaker,
    )
    return (*result, gr.update(value=""))


def _display_ref_audio_name(uploaded_audio: str | None) -> str:
    if uploaded_audio is None or str(uploaded_audio).strip() == "":
        return ""
    return Path(str(uploaded_audio)).name


def _next_available_output_path(
    out_dir: Path,
    base_name: str,
    used_seed: int | str | None,
) -> Path:
    seed_text = "random" if used_seed is None else str(used_seed)
    index = 1
    while True:
        candidate = out_dir / f"{base_name}_{seed_text}_{index:03d}.wav"
        if not candidate.exists():
            return candidate
        index += 1


def _resolve_output_dir(raw_dir: str | None, default_dir: str = DEFAULT_OUTPUT_DIR) -> Path:
    text = "" if raw_dir is None else str(raw_dir).strip()
    if text == "":
        text = default_dir
    return Path(text).expanduser()


def _resolve_internal_output_dir(raw_dir: str | None, default_dir: str = DEFAULT_OUTPUT_DIR) -> Path:
    text = "" if raw_dir is None else str(raw_dir).strip()
    if text == "":
        text = default_dir
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        raise ValueError("Generated WAV Output Directory must be a folder inside the application directory.")
    base_dir = APP_DATA_ROOT
    resolved = (base_dir / candidate).resolve()
    try:
        resolved.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError("Generated WAV Output Directory must stay inside the application directory.") from exc
    return resolved


def _next_available_copy_path(target_dir: Path, filename: str) -> Path:
    candidate = target_dir / filename
    if not candidate.exists():
        return candidate

    source = Path(filename)
    stem = source.stem
    suffix = source.suffix
    index = 1
    while True:
        candidate = target_dir / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _copy_generated_wav(src_path: str | Path, target_dir: Path) -> Path:
    src = Path(src_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        if src.resolve() == target_dir.resolve() / src.name:
            return src
    except Exception:
        pass

    dst = _next_available_copy_path(target_dir, src.name)
    shutil.copy2(src, dst)
    return dst


def _build_runtime_key(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
) -> RuntimeKey:
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(model_device),
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision=str(model_precision),
        codec_device=str(codec_device),
        codec_precision=str(codec_precision),
        enable_watermark=bool(enable_watermark),
        compile_model=False,
        compile_dynamic=False,
    )


def _load_model(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
) -> str:
    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=enable_watermark,
    )
    _, reloaded = get_cached_runtime(runtime_key)
    if reloaded:
        status = "loaded model into memory"
    else:
        status = "model already loaded; reused existing runtime"
    return (
        f"{status}\n"
        f"checkpoint: {runtime_key.checkpoint}\n"
        f"model_device: {runtime_key.model_device}\n"
        f"model_precision: {runtime_key.model_precision}\n"
        f"codec_device: {runtime_key.codec_device}\n"
        f"codec_precision: {runtime_key.codec_precision}"
    )


def _run_generation(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
    text_value: dict | str | None,
    uploaded_audio: str | None,
    output_dir_raw: str,
    save_dir_raw: str,
    num_steps: int,
    num_candidates: int,
    seed_raw: str,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
    speaker_kv_scale_raw: str,
    speaker_kv_min_t_raw: str,
    speaker_kv_max_layers_raw: str,
) -> tuple[object, ...]:
    def stdout_log(msg: str) -> None:
        print(msg, flush=True)

    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=enable_watermark,
    )

    text = _extract_prompt_text(text_value)
    if str(text).strip() == "":
        raise ValueError("text is required.")
    requested_candidates = int(num_candidates)
    if requested_candidates <= 0:
        raise ValueError("num_candidates must be >= 1.")
    if requested_candidates > MAX_GRADIO_CANDIDATES:
        raise ValueError(f"num_candidates must be <= {MAX_GRADIO_CANDIDATES}.")

    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    speaker_kv_min_t = _parse_optional_float(speaker_kv_min_t_raw, "speaker_kv_min_t")
    speaker_kv_max_layers = _parse_optional_int(
        speaker_kv_max_layers_raw,
        "speaker_kv_max_layers",
    )
    seed = _parse_optional_int(seed_raw, "seed")

    ref_wav = _resolve_ref_wav(uploaded_audio=uploaded_audio)
    no_ref = ref_wav is None
    ref_normalize_db = -16.0
    ref_ensure_max = True

    runtime, reloaded = get_cached_runtime(runtime_key)
    stdout_log(f"[gradio] runtime: {'reloaded' if reloaded else 'reused'}")
    stdout_log(
        (
            "[gradio] request: model_device={} model_precision={} codec_device={} codec_precision={} "
            "watermark={} mode={} seconds={} steps={} seed={} no_ref={} candidates={}"
        ).format(
            model_device,
            model_precision,
            codec_device,
            codec_precision,
            enable_watermark,
            cfg_guidance_mode,
            FIXED_SECONDS,
            num_steps,
            "random" if seed is None else seed,
            no_ref,
            requested_candidates,
        )
    )

    result = runtime.synthesize(
        SamplingRequest(
            text=str(text),
            ref_wav=ref_wav,
            ref_latent=None,
            no_ref=bool(no_ref),
            ref_normalize_db=ref_normalize_db,
            ref_ensure_max=bool(ref_ensure_max),
            num_candidates=requested_candidates,
            decode_mode="sequential",
            seconds=FIXED_SECONDS,
            max_ref_seconds=30.0,
            max_text_len=None,
            num_steps=int(num_steps),
            seed=None if seed is None else int(seed),
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_speaker=float(cfg_scale_speaker),
            cfg_scale=cfg_scale,
            cfg_min_t=float(cfg_min_t),
            cfg_max_t=float(cfg_max_t),
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_min_t=speaker_kv_min_t,
            speaker_kv_max_layers=speaker_kv_max_layers,
            trim_tail=True,
        ),
        log_fn=stdout_log,
    )

    _save_app_settings(output_dir=output_dir_raw, save_dir=save_dir_raw)

    out_dir = _resolve_internal_output_dir(output_dir_raw, default_dir=DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if uploaded_audio:
        base_name = Path(uploaded_audio).stem
    else:
        base_name = "no_ref"

    out_paths: list[str] = []
    for audio in result.audios:
        out_path = _next_available_output_path(
            out_dir=out_dir,
            base_name=base_name,
            used_seed=result.used_seed,
        )
        out_path = save_wav(
            out_path,
            audio.float(),
            result.sample_rate,
        )

        _embed_wav_metadata(
            out_path,
            text,
            num_steps,
            num_candidates,
            result.used_seed,
            cfg_guidance_mode,
            cfg_scale_text,
            cfg_scale_speaker,
        )

        out_paths.append(str(out_path))

    runtime_msg = "runtime: reloaded" if reloaded else "runtime: reused"
    detail_lines = [
        runtime_msg,
        f"seed_used: {result.used_seed}",
        f"candidates: {len(result.audios)}",
        f"output_dir: {out_dir}",
        *[f"saved[{i}]: {path}" for i, path in enumerate(out_paths, start=1)],
    ]
    detail_lines.extend(result.messages)
    detail_text = "\n".join(detail_lines)
    timing_text = _format_timings(result.stage_timings, result.total_to_decode)
    stdout_log(f"[gradio] saved {len(out_paths)} candidates")

    slot_updates: list[object] = []
    audio_updates: list[object] = []
    save_button_updates: list[object] = []
    for i in range(MAX_GRADIO_CANDIDATES):
        if i < len(out_paths):
            slot_updates.append(gr.update(visible=True))
            audio_updates.append(gr.update(value=out_paths[i], visible=True))
            save_button_updates.append(gr.update(visible=True))
        else:
            slot_updates.append(gr.update(visible=False))
            audio_updates.append(gr.update(value=None, visible=False))
            save_button_updates.append(gr.update(visible=False))
    return (
        *slot_updates,
        *audio_updates,
        *save_button_updates,
        detail_text,
        timing_text,
        str(result.used_seed),
    )


def _clear_runtime_cache() -> str:
    clear_cached_runtime()
    return "cleared loaded model from memory"


def _clear_seed_value() -> str:
    return ""


def _clear_prompt_value() -> dict[str, object]:
    return {"text": "", "files": []}


def _restore_last_seed_value(last_seed_value: str | None) -> str:
    if last_seed_value is None:
        return ""
    return str(last_seed_value).strip()


def _set_generate_buttons_busy() -> tuple[gr.Button, gr.Button]:
    return (
        gr.update(value="Generating...", variant="stop", interactive=False),
        gr.update(value="Generating...", variant="stop", interactive=False),
    )


def _set_generate_buttons_ready() -> tuple[gr.Button, gr.Button]:
    return (
        gr.update(value="Generate", variant="primary", interactive=True),
        gr.update(value="Generate", variant="primary", interactive=True),
    )


def _set_save_button_busy() -> gr.Button:
    return gr.update(value="Saving...", variant="primary", interactive=False)


def _set_save_button_ready() -> gr.Button:
    return gr.update(value="Save", variant="primary", interactive=True)


def _save_generated_audio(
    generated_audio_path: str | None,
    output_dir_raw: str,
    save_dir_raw: str,
    current_log: str,
    audio_index: int,
) -> str:
    if generated_audio_path is None or str(generated_audio_path).strip() == "":
        raise ValueError(f"Generated Audio {audio_index} is empty.")

    save_dir_text = str(save_dir_raw or "").strip()
    if save_dir_text == "":
        raise ValueError("Generated WAV Save Directory is blank.")

    _save_app_settings(output_dir=output_dir_raw, save_dir=save_dir_raw)

    save_dir = _resolve_output_dir(save_dir_text, default_dir="")
    saved_path = _copy_generated_wav(generated_audio_path, save_dir)

    lines = [str(current_log or "").rstrip()] if str(current_log or "").strip() else []
    lines.append(f"saved_manual[{audio_index}]: {saved_path}")
    return "\n".join(lines)


def _load_reference_preset_ui_state() -> tuple[object, ...]:
    app_settings = _load_app_settings()
    presets, changed = _hydrate_reference_preset_prompt_previews(
        app_settings.get("reference_presets")
    )
    if changed:
        _save_app_settings(reference_presets=presets)
    selected_slot_index = None
    uploaded_audio = None
    ref_audio_name = ""

    button_updates = [
        _reference_preset_button_update(preset, index)
        for index, preset in enumerate(presets)
    ]
    clear_updates = [
        _reference_preset_clear_button_update(preset)
        for preset in presets
    ]
    color_updates = [
        _reference_preset_color_button_update(preset, index)
        for index, preset in enumerate(presets)
    ]

    return (
        *[_reference_preset_prompt_preview_update(preset) for preset in presets],
        *[gr.update(value=str(preset.get("path", "") or "").strip()) for preset in presets],
        *button_updates,
        *clear_updates,
        *color_updates,
        presets,
        selected_slot_index,
        _build_reference_preset_style_html(presets, selected_slot_index),
        gr.update(value=uploaded_audio),
        ref_audio_name,
        _reference_preset_notice_update(),
    )


def build_ui() -> gr.Blocks:
    app_settings = _load_app_settings()
    loaded_reference_presets, changed = _hydrate_reference_preset_prompt_previews(
        app_settings.get("reference_presets")
    )
    if changed:
        _save_app_settings(reference_presets=loaded_reference_presets)
    loaded_active_reference_preset_slot = None
    initial_uploaded_audio = None
    initial_ref_audio_name = ""
    default_checkpoint = _default_checkpoint()
    default_model_device = _default_model_device()
    default_codec_device = _default_codec_device()
    device_choices = list_available_runtime_devices()
    model_precision_choices = _precision_choices_for_device(default_model_device)
    codec_precision_choices = _precision_choices_for_device(default_codec_device)

    with gr.Blocks(title="Irodori-TTS Gradio") as demo:
        gr.Markdown("# Irodori-TTS Inference (Cached Runtime)")
        gr.Markdown(
            "When settings are unchanged, runtime is reused and only sampling/decoding runs."
        )

        with gr.Row():
            checkpoint = gr.Textbox(
                label="Checkpoint (.pt/.safetensors or HF repo id)",
                value=default_checkpoint,
                scale=4,
            )
            model_device = gr.Dropdown(
                label="Model Device",
                choices=device_choices,
                value=default_model_device,
                scale=1,
            )
            model_precision = gr.Dropdown(
                label="Model Precision",
                choices=model_precision_choices,
                value=model_precision_choices[0],
                scale=1,
            )
            codec_device = gr.Dropdown(
                label="Codec Device",
                choices=device_choices,
                value=default_codec_device,
                scale=1,
            )
            codec_precision = gr.Dropdown(
                label="Codec Precision",
                choices=codec_precision_choices,
                value=codec_precision_choices[0],
                scale=1,
            )
            enable_watermark = gr.State(False)
            last_seed_value = gr.State("")
            reference_preset_state = gr.State(loaded_reference_presets)
            selected_reference_preset_slot = gr.State(loaded_active_reference_preset_slot)
            preserve_reference_preset_selection = gr.State(False)

        with gr.Row():
            load_model_btn = gr.Button("Load Model")
            clear_cache_btn = gr.Button("Unload Model")
            clear_cache_msg = gr.Textbox(label="Model Status", interactive=False)

        with gr.Accordion("File Output Settings", open=False):
            output_dir_raw = gr.Textbox(
                label="Generated WAV Output Folder (inside app folder only)",
                value=app_settings.get("output_dir", DEFAULT_OUTPUT_DIR),
            )
            save_dir_raw = gr.Textbox(
                label="Generated WAV Save Directory (full path OK)",
                value=app_settings.get("save_dir", ""),
            )
            gr.Markdown(
                "Generated WAVs are created in a folder inside the app directory. Press the small Save button under each Generated Audio player to copy that WAV into the save directory."
            )

        reference_preset_style_html = gr.HTML(
            _build_reference_preset_style_html(
                loaded_reference_presets, loaded_active_reference_preset_slot
            ),
            elem_id=REFERENCE_PRESET_STYLE_HTML_ID,
        )

        reference_preset_uploaders: list[gr.File] = []
        reference_preset_buttons: list[gr.Button] = []
        reference_preset_clear_buttons: list[gr.Button] = []
        reference_preset_color_buttons: list[gr.HTML] = []
        reference_preset_prompt_previews: list[gr.HTML] = []
        reference_preset_color_values: list[gr.Textbox] = []
        reference_preset_path_values: list[gr.Textbox] = []
        reference_preset_reorder_value = gr.Textbox(
            value="",
            visible=True,
            container=False,
            elem_id=REFERENCE_PRESET_REORDER_INPUT_ID,
            elem_classes=["reference-preset-reorder-value"],
        )
        preset_metadata_drop_value = gr.Textbox(
            value="",
            visible=True,
            container=False,
            elem_id=PRESET_METADATA_DROP_INPUT_ID,
            elem_classes=["preset-metadata-drop-value"],
        )

        with gr.Group():
            gr.Markdown("### Reference Audio Presets")
            gr.Markdown("Drag a WAV onto a slot, then press the preset button to set it as Reference Audio.")
            reference_preset_notice = gr.HTML(
                "",
                visible=True,
                elem_id=REFERENCE_PRESET_NOTICE_ID,
            )
            for preset_row in range(REFERENCE_PRESET_ROWS):
                with gr.Row():
                    for preset_col in range(REFERENCE_PRESET_COLS):
                        preset_index = preset_row * REFERENCE_PRESET_COLS + preset_col
                        preset = loaded_reference_presets[preset_index]
                        with gr.Column(
                            min_width=118,
                            elem_classes=["reference-preset-slot"],
                            elem_id=f"{REFERENCE_PRESET_SLOT_ID_PREFIX}{preset_index}",
                        ):
                            uploader = gr.File(
                                label="Drop WAV",
                                file_types=[".wav"],
                                file_count="single",
                                type="filepath",
                                elem_classes=[REFERENCE_PRESET_UPLOAD_CLASS],
                            )
                            color_value = gr.Textbox(
                                value=_normalize_reference_preset_color(preset.get("color")),
                                visible=True,
                                container=False,
                                elem_id=f"{REFERENCE_PRESET_COLOR_INPUT_ID_PREFIX}{preset_index}",
                                elem_classes=["reference-preset-color-value"],
                            )
                            path_value = gr.Textbox(
                                value=str(preset.get("path", "") or "").strip(),
                                visible=True,
                                container=False,
                                elem_id=f"{REFERENCE_PRESET_PATH_INPUT_ID_PREFIX}{preset_index}",
                                elem_classes=["reference-preset-path-value"],
                            )
                            prompt_preview = gr.HTML(
                                _reference_preset_prompt_preview_html(preset),
                                visible=True,
                                container=False,
                            )
                            with gr.Row(elem_classes=["reference-preset-button-row"]):
                                button = gr.Button(
                                    _reference_preset_button_label(preset, preset_index),
                                    variant="secondary",
                                    interactive=_reference_preset_has_file(preset),
                                    elem_id=f"{REFERENCE_PRESET_BUTTON_ID_PREFIX}{preset_index}",
                                    elem_classes=[REFERENCE_PRESET_SET_BUTTON_CLASS],
                                    scale=1,
                                )
                                with gr.Column(elem_classes=["reference-preset-button-tools"], scale=0, min_width=18):
                                    clear_button = gr.Button(
                                        "×",
                                        variant="secondary",
                                        interactive=_reference_preset_has_file(preset),
                                        elem_classes=[REFERENCE_PRESET_CLEAR_BUTTON_CLASS],
                                        scale=0,
                                        min_width=18,
                                    )
                                    color_button = gr.HTML(
                                        _reference_preset_color_picker_html(preset, preset_index),
                                        visible=True,
                                        container=False,
                                        elem_classes=["reference-preset-color-host"],
                                    )
                            reference_preset_uploaders.append(uploader)
                            reference_preset_buttons.append(button)
                            reference_preset_clear_buttons.append(clear_button)
                            reference_preset_color_buttons.append(color_button)
                            reference_preset_prompt_previews.append(prompt_preview)
                            reference_preset_color_values.append(color_value)
                            reference_preset_path_values.append(path_value)

        with gr.Row():
            uploaded_audio = gr.Audio(
                label="Reference Audio Upload (optional, blank = no-reference mode)",
                type="filepath",
                value=initial_uploaded_audio,
                scale=3,
            )
            ref_audio_name = gr.Textbox(
                label="Reference Audio File Name",
                interactive=False,
                value=initial_ref_audio_name,
                scale=1,
            )

        with gr.Row(elem_id=TEXT_EMOJI_ROW_ID):
            with gr.Column(scale=8, elem_id=PROMPT_COLUMN_ID):
                with gr.Row(elem_id=PROMPT_ACTIONS_ROW_ID):
                    with gr.Column(scale=0, min_width=0):
                        prompt_clear_btn = gr.Button(
                            "Clear Text",
                            variant="secondary",
                            elem_id=PROMPT_CLEAR_BUTTON_ID,
                            scale=0,
                            min_width=0,
                        )
                text = gr.MultimodalTextbox(
                    label="Text",
                    lines=6,
                    max_lines=10,
                    file_count="single",
                    file_types=["audio"],
                    placeholder="Type text here, or drop a WAV with embedded metadata onto this box.",
                    elem_id=PROMPT_BOX_ID,
                )
                prompt_generate_btn = gr.Button(
                    "Generate",
                    variant="primary",
                    elem_id=PROMPT_GENERATE_BUTTON_ID,
                )
            with gr.Column(scale=4, elem_id=EMOJI_COLUMN_ID):
                gr.HTML(_build_emoji_panel_html(), elem_id=EMOJI_HTML_ID)

        with gr.Accordion("Sampling", open=True):
            with gr.Row():
                num_steps = gr.Slider(label="Num Steps", minimum=1, maximum=120, value=40, step=1)
                num_candidates = gr.Slider(
                    label="Num Candidates",
                    minimum=1,
                    maximum=MAX_GRADIO_CANDIDATES,
                    value=1,
                    step=1,
                )
                with gr.Column(scale=1, elem_id="seed-input-wrap", min_width=220):
                    gr.HTML('<div id="seed-label">Seed (blank=random)</div>')
                    with gr.Row(elem_id="seed-input-row"):
                        seed_raw = gr.Textbox(
                            label=None,
                            value="",
                            scale=1,
                            min_width=0,
                            container=False,
                            elem_id=SEED_TEXTBOX_ID,
                        )
                        seed_clear_btn = gr.Button(
                            "Clear",
                            variant="secondary",
                            scale=0,
                            min_width=56,
                            elem_id=SEED_CLEAR_BUTTON_ID,
                        )
                        seed_last_btn = gr.Button(
                            "Last",
                            variant="secondary",
                            scale=0,
                            min_width=56,
                            elem_id=SEED_LAST_BUTTON_ID,
                        )

            with gr.Row():
                cfg_guidance_mode = gr.Dropdown(
                    label="CFG Guidance Mode",
                    choices=["independent", "joint", "alternating"],
                    value="independent",
                )
                cfg_scale_text = gr.Slider(
                    label="CFG Scale Text",
                    minimum=0.0,
                    maximum=10.0,
                    value=3.0,
                    step=0.1,
                )
                cfg_scale_speaker = gr.Slider(
                    label="CFG Scale Speaker",
                    minimum=0.0,
                    maximum=10.0,
                    value=5.0,
                    step=0.1,
                )

        with gr.Accordion("Advanced (Optional)", open=False):
            cfg_scale_raw = gr.Textbox(label="CFG Scale Override (optional)", value="")
            with gr.Row():
                cfg_min_t = gr.Number(label="CFG Min t", value=0.5)
                cfg_max_t = gr.Number(label="CFG Max t", value=1.0)
                context_kv_cache = gr.Checkbox(label="Context KV Cache", value=True)
            with gr.Row():
                truncation_factor_raw = gr.Textbox(label="Truncation Factor (optional)", value="")
                rescale_k_raw = gr.Textbox(label="Rescale k (optional)", value="")
                rescale_sigma_raw = gr.Textbox(label="Rescale sigma (optional)", value="")
            with gr.Row():
                speaker_kv_scale_raw = gr.Textbox(label="Speaker KV Scale (optional)", value="")
                speaker_kv_min_t_raw = gr.Textbox(label="Speaker KV Min t (optional)", value="0.9")
                speaker_kv_max_layers_raw = gr.Textbox(
                    label="Speaker KV Max Layers (optional)", value=""
                )

        main_generate_btn = gr.Button(
            "Generate",
            variant="primary",
            elem_id=MAIN_GENERATE_BUTTON_ID,
        )

        out_audio_slots: list[gr.Column] = []
        out_audios: list[gr.Audio] = []
        save_generated_audio_buttons: list[gr.Button] = []
        num_rows = (
            MAX_GRADIO_CANDIDATES + GRADIO_AUDIO_COLS_PER_ROW - 1
        ) // GRADIO_AUDIO_COLS_PER_ROW
        with gr.Column():
            for row_idx in range(num_rows):
                with gr.Row():
                    for col_idx in range(GRADIO_AUDIO_COLS_PER_ROW):
                        i = row_idx * GRADIO_AUDIO_COLS_PER_ROW + col_idx
                        if i >= MAX_GRADIO_CANDIDATES:
                            break
                        with gr.Column(min_width=160, visible=(i == 0)) as audio_slot:
                            audio_component = gr.Audio(
                                label=f"Generated Audio {i + 1}",
                                type="filepath",
                                interactive=False,
                                visible=(i == 0),
                                min_width=160,
                            )
                            out_audio_slots.append(audio_slot)
                            out_audios.append(audio_component)
                            save_button = gr.Button(
                                "Save",
                                variant="primary",
                                visible=(i == 0),
                                elem_classes=[SAVE_GENERATED_AUDIO_BUTTON_CLASS],
                            )
                            save_generated_audio_buttons.append(save_button)
        out_log = gr.Textbox(label="Run Log", lines=8)
        out_timing = gr.Textbox(label="Timing", lines=8)

        demo.load(
            _load_reference_preset_ui_state,
            outputs=[
                *reference_preset_prompt_previews,
                *reference_preset_path_values,
                *reference_preset_buttons,
                *reference_preset_clear_buttons,
                *reference_preset_color_buttons,
                reference_preset_state,
                selected_reference_preset_slot,
                reference_preset_style_html,
                uploaded_audio,
                ref_audio_name,
                reference_preset_notice,
            ],
            queue=False,
        )

        text.change(
            _load_prompt_metadata_from_box,
            inputs=[
                text,
                num_steps,
                num_candidates,
                seed_raw,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
            ],
            outputs=[
                text,
                num_steps,
                num_candidates,
                seed_raw,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
            ],
            queue=False,
        )
        text.input(
            _load_prompt_metadata_from_box,
            inputs=[
                text,
                num_steps,
                num_candidates,
                seed_raw,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
            ],
            outputs=[
                text,
                num_steps,
                num_candidates,
                seed_raw,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
            ],
            queue=False,
        )
        preset_metadata_drop_value.change(
            _load_prompt_metadata_from_preset_path,
            inputs=[
                preset_metadata_drop_value,
                text,
                num_steps,
                num_candidates,
                seed_raw,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
            ],
            outputs=[
                text,
                num_steps,
                num_candidates,
                seed_raw,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
                preset_metadata_drop_value,
            ],
            queue=False,
        )
        uploaded_audio.change(
            _handle_reference_audio_change,
            inputs=[uploaded_audio, reference_preset_state, preserve_reference_preset_selection],
            outputs=[
                ref_audio_name,
                selected_reference_preset_slot,
                reference_preset_style_html,
                preserve_reference_preset_selection,
                reference_preset_notice,
            ],
            queue=False,
        )

        output_dir_raw.change(
            _persist_directory_settings,
            inputs=[output_dir_raw, save_dir_raw],
            queue=False,
        )
        save_dir_raw.change(
            _persist_directory_settings,
            inputs=[output_dir_raw, save_dir_raw],
            queue=False,
        )

        for i, preset_uploader in enumerate(reference_preset_uploaders):
            preset_uploader.change(
                partial(_register_reference_preset, slot_index=i),
                inputs=[
                    preset_uploader,
                    output_dir_raw,
                    save_dir_raw,
                    reference_preset_state,
                    selected_reference_preset_slot,
                ],
                outputs=[
                    reference_preset_uploaders[i],
                    reference_preset_path_values[i],
                    reference_preset_buttons[i],
                    reference_preset_clear_buttons[i],
                    reference_preset_color_buttons[i],
                    reference_preset_prompt_previews[i],
                    reference_preset_state,
                    reference_preset_style_html,
                    selected_reference_preset_slot,
                    reference_preset_notice,
                ],
                queue=False,
            )
            reference_preset_buttons[i].click(
                partial(_apply_reference_preset, slot_index=i),
                inputs=[reference_preset_state, uploaded_audio, selected_reference_preset_slot],
                outputs=[
                    uploaded_audio,
                    ref_audio_name,
                    selected_reference_preset_slot,
                    reference_preset_style_html,
                    preserve_reference_preset_selection,
                    reference_preset_notice,
                ],
                queue=False,
            )
            reference_preset_clear_buttons[i].click(
                partial(_clear_reference_preset, slot_index=i),
                inputs=[
                    uploaded_audio,
                    output_dir_raw,
                    save_dir_raw,
                    reference_preset_state,
                    selected_reference_preset_slot,
                ],
                outputs=[
                    reference_preset_path_values[i],
                    reference_preset_buttons[i],
                    reference_preset_clear_buttons[i],
                    reference_preset_color_buttons[i],
                    reference_preset_prompt_previews[i],
                    reference_preset_state,
                    reference_preset_style_html,
                    selected_reference_preset_slot,
                    uploaded_audio,
                    ref_audio_name,
                    reference_preset_notice,
                ],
                queue=False,
            )
            reference_preset_color_values[i].change(
                partial(_set_reference_preset_color, slot_index=i),
                inputs=[
                    reference_preset_color_values[i],
                    output_dir_raw,
                    save_dir_raw,
                    reference_preset_state,
                    selected_reference_preset_slot,
                ],
                outputs=[
                    reference_preset_buttons[i],
                    reference_preset_state,
                    reference_preset_style_html,
                    selected_reference_preset_slot,
                    reference_preset_notice,
                ],
                queue=False,
            )

        reference_preset_reorder_value.change(
            _swap_reference_presets,
            inputs=[
                reference_preset_reorder_value,
                output_dir_raw,
                save_dir_raw,
                reference_preset_state,
                selected_reference_preset_slot,
            ],
            outputs=[
                *reference_preset_prompt_previews,
                *reference_preset_path_values,
                *reference_preset_color_values,
                *reference_preset_buttons,
                *reference_preset_clear_buttons,
                *reference_preset_color_buttons,
                reference_preset_state,
                selected_reference_preset_slot,
                reference_preset_style_html,
                uploaded_audio,
                ref_audio_name,
                reference_preset_notice,
                reference_preset_reorder_value,
            ],
            queue=False,
        )

        seed_clear_btn.click(_clear_seed_value, outputs=[seed_raw], queue=False)
        prompt_clear_btn.click(_clear_prompt_value, outputs=[text], queue=False)
        seed_last_btn.click(
            _restore_last_seed_value,
            inputs=[last_seed_value],
            outputs=[seed_raw],
            queue=False,
        )

        for i, save_button in enumerate(save_generated_audio_buttons):
            save_click = save_button.click(
                _set_save_button_busy,
                outputs=[save_button],
                queue=False,
            )
            save_click = save_click.then(
                partial(_save_generated_audio, audio_index=i + 1),
                inputs=[out_audios[i], output_dir_raw, save_dir_raw, out_log],
                outputs=[out_log],
                queue=False,
            )
            save_click.then(
                _set_save_button_ready,
                outputs=[save_button],
                queue=False,
            )

        prompt_generate_event = prompt_generate_btn.click(
            _set_generate_buttons_busy,
            outputs=[prompt_generate_btn, main_generate_btn],
            queue=False,
        )
        prompt_generate_event = prompt_generate_event.then(
            _run_generation,
            inputs=[
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                enable_watermark,
                text,
                uploaded_audio,
                output_dir_raw,
                save_dir_raw,
                num_steps,
                num_candidates,
                seed_raw,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
                cfg_scale_raw,
                cfg_min_t,
                cfg_max_t,
                context_kv_cache,
                truncation_factor_raw,
                rescale_k_raw,
                rescale_sigma_raw,
                speaker_kv_scale_raw,
                speaker_kv_min_t_raw,
                speaker_kv_max_layers_raw,
            ],
            outputs=[*out_audio_slots, *out_audios, *save_generated_audio_buttons, out_log, out_timing, last_seed_value],
        )
        prompt_generate_event.then(
            _set_generate_buttons_ready,
            outputs=[prompt_generate_btn, main_generate_btn],
            queue=False,
        )

        main_generate_event = main_generate_btn.click(
            _set_generate_buttons_busy,
            outputs=[prompt_generate_btn, main_generate_btn],
            queue=False,
        )
        main_generate_event = main_generate_event.then(
            _run_generation,
            inputs=[
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                enable_watermark,
                text,
                uploaded_audio,
                output_dir_raw,
                save_dir_raw,
                num_steps,
                num_candidates,
                seed_raw,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
                cfg_scale_raw,
                cfg_min_t,
                cfg_max_t,
                context_kv_cache,
                truncation_factor_raw,
                rescale_k_raw,
                rescale_sigma_raw,
                speaker_kv_scale_raw,
                speaker_kv_min_t_raw,
                speaker_kv_max_layers_raw,
            ],
            outputs=[*out_audio_slots, *out_audios, *save_generated_audio_buttons, out_log, out_timing, last_seed_value],
        )
        main_generate_event.then(
            _set_generate_buttons_ready,
            outputs=[prompt_generate_btn, main_generate_btn],
            queue=False,
        )
        model_device.change(
            _on_model_device_change, inputs=[model_device], outputs=[model_precision]
        )
        codec_device.change(
            _on_codec_device_change, inputs=[codec_device], outputs=[codec_precision]
        )

        load_model_btn.click(
            _load_model,
            inputs=[
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                enable_watermark,
            ],
            outputs=[clear_cache_msg],
        )
        clear_cache_btn.click(_clear_runtime_cache, outputs=[clear_cache_msg])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio app for Irodori-TTS with cached runtime.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
        css=CUSTOM_CSS,
        head=CUSTOM_HEAD,
    )


if __name__ == "__main__":
    main()
