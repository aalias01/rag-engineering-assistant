# Capture Demo Media

Repeatable process for producing the README screenshots and a short demo GIF after the app is running.

## Prerequisites

- Final corpus is in `data/documents/` and has been ingested (`/health` returns `chroma_loaded: true`).
- API is running on `http://localhost:8000`.
- Frontend `frontend/index.html` is open in a browser at a reasonable zoom (100%) and window width (~1280px).
- Browser console shows no errors.

## Shot List

Pick three real corpus-grounded questions before starting. One should produce a clean grounded answer, one should exercise multi-page citations, one should be deliberately out-of-corpus (refusal).

| # | Filename | What it shows |
|---|----------|---------------|
| 1 | `figures/demo_chat.png` | Header (Ready badge + chunk count), the input area, and a clean grounded answer in the conversation. |
| 2 | `figures/source_citations.png` | The sources panel with at least two distinct document/page pills. |
| 3 | `figures/retrieved_chunks.png` | The retrieved-chunks accordion with at least one chunk expanded showing actual document text. |
| 4 (optional) | `figures/api_docs.png` | `http://localhost:8000/docs` with `/query` expanded. |
| 5 (optional) | `figures/refusal.png` | An out-of-corpus question producing a clean refusal. |

## Capture Steps (macOS)

1. Reset the conversation by reloading `frontend/index.html`.
2. Click an example query button or type the question.
3. Wait for the answer to finish streaming and the sources panel to populate.
4. Take a window screenshot:

   ```text
   Cmd + Shift + 4, then Space, then click the browser window
   ```

   Save with the filename in the table above.
5. Move the file into `figures/` at the repo root.

## Capture Steps (Linux/Windows)

- Linux: `gnome-screenshot -w` or `flameshot gui`.
- Windows: Snipping Tool window-mode (Win + Shift + S → window).

## Optional Demo GIF (60-90 seconds)

Recommended on macOS with [Kap](https://getkap.co/) (free, native). Other options: LICEcap, ScreenToGif, OBS + ffmpeg.

Script:

1. Open frontend on a clean conversation.
2. Type one in-corpus question. Let it stream.
3. Click a source pill or expand a chunk.
4. Reset, then type one out-of-corpus question to show refusal.
5. Stop recording, export as `figures/demo.gif`, target file size < 5 MB.

GIF settings:

- 1280×720 or smaller
- 12-15 fps
- Loop on

## File Hygiene

Before committing:

- Crop chrome (browser address bar, dock, system clock) out of every screenshot.
- Use a light-theme background unless the README explicitly uses dark.
- Confirm no real API keys, personal email, or private tabs are visible in the chrome.
- File sizes: PNG < 500 KB each, GIF < 5 MB.

## README Linkage

After the files are saved under `figures/`, add (or update) the README image block:

```markdown
![Chat demo](figures/demo_chat.png)
![Source citations](figures/source_citations.png)
![Retrieved chunks](figures/retrieved_chunks.png)
```

If a GIF is included, put it near the top of the README under the live-demo links.
