# GestureX Demo

This directory is reserved for a live demonstration video, GIF, or screenshot showing the real-time deployment application in action.

## What to record

After collecting data, training, and running `python deployment/app.py`, capture a short screen recording of the webcam window showing:

- Detected hand landmarks and bounding box
- Gesture label (e.g., **Victory**, **Thumbs Up**)
- Confidence percentage (e.g., 96.3%)
- Live FPS counter
- The `UNKNOWN` state when no hand is present or confidence is below threshold

## Suggested tools

| Platform | Tool |
| --- | --- |
| Windows | Xbox Game Bar (`Win+G` → Capture), OBS Studio, ShareX |
| macOS | QuickTime Player, OBS Studio |
| Linux | OBS Studio, `ffmpeg` |

Convert to GIF for embedding in the README:

```bash
ffmpeg -i demo.mp4 -vf "fps=12,scale=640:-1" -loop 0 demo.gif
```

## Embedding in README

Add the following line to the `README.md` key-findings or results section after capturing:

```markdown
![GestureX real-time demo](demo/demo.gif)
```

> **Privacy reminder:** do not include identifiable video of people without explicit consent. Blur or crop the face before uploading to a public repository.
