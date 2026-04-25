# Irodori-TTS Gradio UI Extended

このリポジトリは [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) を元にした fork / 改変版です。

Irodori-TTS を使いやすくすることを目的にしています。
主に、参照音声プリセット、生成時の設定復元、絵文字パネルなどを追加しています。

## Screenshots

![Reference Audio Presets](docs/images/reference-presets.png)

![Generation UI](docs/images/generation-ui.png)

## Installation

```bash
git clone https://github.com/tgyou2/Irodori-TTS-Gradio-Extended.git
cd Irodori-TTS-Gradio-Extended
uv sync
```

## Gradio Web UI

```bash
uv run python gradio_app.py --server-name 127.0.0.1 --server-port 7860
```

## 主な追加機能

### File Output Settings

生成 WAV ファイルの出力先フォルダと保存先フォルダを UI から設定できるようにしました。

作業用の出力先フォルダと、採用したい WAV ファイルの保存先フォルダを分けて管理できます。

### Reference Audio Presets

参照音声ファイルをプリセットボタンへ登録し、ワンクリックで呼び出せるようにしました。

プリセットボタンは色を変更できます。  
また、ドラッグ操作で好きな位置へ並べ替えられます。

WAV ファイルに音声テキストのメタデータが埋め込まれている場合は、プリセット枠内にその内容の一部を表示します。

動作確認用のサンプルとして、`examples/reference_audio_presets_test.wav` を同梱しています。

### Reference Audio File Name

現在読み込まれている参照 WAV ファイル名を表示するようにしました。

### 生成 WAV ファイル名の変更

生成した WAV ファイル名に、使用した参照 WAV ファイル名を反映するようにしました。

生成した WAV ファイルを見た時点で、どの参照 WAV ファイルを使ったものか判別しやすくなります。

### 生成 WAV ファイルへのメタデータ付与

生成した WAV ファイルに、生成時の設定をメタデータとして付与する機能を追加しました。

保存される主な情報は以下です。

- 音声テキスト
- Num Steps
- Num Candidates
- Seed
- CFG Guidance Mode
- CFG Scale Text
- CFG Scale Speaker

メタデータは通常のファイル操作では見えにくい情報ですが、後述の「生成時の設定復元」機能で使用します。

この fork では [Mutagen](https://mutagen.readthedocs.io/) を依存関係に含めています。

### Mutagen 依存について

メタデータ付与のために、タグ読み書きを行うライブラリ Mutagen に依存しています。

Mutagen は Python 製のメタデータ操作ライブラリです。  
出力するファイルにタイトルや任意情報を書き込んだり、既存のタグを読み出したりする用途に使われます。

この fork では、生成した WAV ファイルに対して ID3 タグ形式で生成時の設定を書き込み、再読み込み時にその情報を取得します。

### 生成時の設定復元

メタデータ付き WAV ファイルを Text 欄へドロップすると、埋め込まれた内容を読み取って、生成時の設定を復元できます。

また、プリセットボタンを Text 欄へドラッグ＆ドロップすることで、その参照 WAV ファイルに埋め込まれた内容を復元することもできます。

過去に生成した生成時の設定を再利用したいときに便利です。

### 絵文字 / 記号パネル

感情表現や発声補助用の記号を、ボタンから入力できるようにしました。

### Clear Text

Text 欄の内容を空にする Clear Text ボタンを追加しました。

### Seed の Clear / Last

Seed 欄に Clear と Last を追加しました。

ランダム生成に戻す操作と、直前の Seed を再利用する操作をしやすくしました。

### Generate ボタンの追加

Generate ボタンをText 欄の下にも配置しました。

### Save ボタン

生成した WAV ファイルに個別の Save ボタンを追加しました。

採用したい WAV ファイルだけをワンクリックで保存先フォルダへコピーできます。

### Generate / Save ボタンの視認性改善

Generate ボタンおよび Save ボタンについて、押したことや処理中であることが分かりやすいように見た目を調整しました。

### 生成時のモデル取得の挙動

生成時にモデル取得関連で通信エラーがよく起こっていたため、
モデルを一度ダウンロードしたあとはローカルキャッシュを使用するようにしました。

配布元のモデルが更新されても自動では反映されないため、最新版を使いたい場合はキャッシュを削除して再取得するか、ローカルのモデルファイルを手動で更新してください。

## 保存されるローカルファイルについて

この fork は、設定やプリセットをローカルファイルとして保存します。

主な保存先は以下です。

- `gradio_app_settings.json`
- `reference_audio_presets/`
- `output_voice/`

これらには、個人の設定、参照 WAV ファイル、生成した WAV ファイルが含まれる可能性があります。  
GitHub へ公開する場合は、これらをコミットしないように注意してください。

`.gitignore` には、少なくとも以下を含めることをおすすめします。

```gitignore
gradio_app_settings.json
reference_audio_presets/
output_voice/
.gradio_visible_outputs/
__pycache__/
*.pyc
.venv/
venv/
```

## ライセンスと利用条件

この fork は [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) を元にした Gradio UI 改変版です。

コード、モデル重み、学習済みモデル、音声サンプルなどの利用条件は、元プロジェクトおよび各配布元のライセンスを確認してください。

この fork によって、元モデルや元プロジェクトのライセンス条件が変更されることはありません。

また、本人の同意なしに実在人物の声を模倣する用途、なりすまし、誤情報、権利侵害につながる用途には使用しないでください。

## Credits

この fork は [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) を元にしています。  
元プロジェクトの作者および関係者に感謝します。

この fork では、主に Gradio UI の操作性改善と、生成作業を繰り返し行うための補助機能を追加しています。

利用している主なライブラリ・ツール:

- [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
- [Gradio](https://www.gradio.app/)
- [Mutagen](https://mutagen.readthedocs.io/)


