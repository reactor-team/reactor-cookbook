# Preset portraits

**Portrait image files are deliberately not committed to this repo.** The
example ships the three preset _recipes_ (script, prompt, wpm, seed) but not
the faces they run on, because a face in a public repo is a licensing and
consent question, not a code question.

Until a file exists, the preset row shows the persona's monogram and
clicking it applies everything except the image, so the example still runs and
still demonstrates the command sequence. Upload your own face in the take panel
and press Start.

## Adding portraits

Drop files here named after the preset id:

| File            | Preset          |
| --------------- | --------------- |
| `teddy.jpg`     | Teddy Bear      |
| `announcer.jpg` | Radio Announcer |
| `grandma.jpg`   | Grandma         |

Then remove the `public/presets/*.jpg` line from the example's `.gitignore` if
you intend to commit them.

Whatever you add has to satisfy the sourcing policy below, and the provenance
table is the record that it does.

## Sourcing policy

**No real-person or celebrity likenesses.** This model animates a face and puts
words in its mouth; running it on someone's likeness without their consent is
not something a public example should demonstrate. Acceptable sources:

- Fully synthetic, AI-generated faces from a dataset whose licence permits
  redistribution (record the dataset and licence below).
- A photo of yourself, or of someone who has explicitly agreed to it.

Record the provenance of anything you add:

| File | Source | Licence |
| ---- | ------ | ------- |
|      |        |         |

## Framing spec

A single, well-lit person facing the camera. Beyond that, **framing is the
thing that most decides whether output looks good**, and it is easy to get
wrong in a way that is invisible until you run a take.

The model fits the image to its 640×352 canvas. That is a very wide frame
(1.82:1), much wider than a normal portrait. Feed it a square headshot where
the head fills the frame and the fit produces an extreme facial close-up with
the top of the head cut off. Cropping cannot rescue such an image, because
there is no margin to crop into: the fix has to happen when the image is
sourced.

So source portraits **already wide**:

| Property   | Target                                                      |
| ---------- | ----------------------------------------------------------- |
| Aspect     | 1.82:1 (640×352), or wider and croppable to it              |
| Resolution | ≥ 1280×704, so the crop step has pixels to work with        |
| Subject    | Head and shoulders, occupying the centre third horizontally |
| Headroom   | Clear space above the hair — hair must not touch the edge   |
| Background | Flat and uncluttered; the model holds it steadier           |

A quick check before you commit one: crop it to 640×352 and look at it. If you
had to cut the hair or the chin to get there, the source is wrong, however good
the photo is.

Uploads from users go through the app's crop step
(`app/components/CropModal.tsx`), which lets them frame the region themselves
and uses the browser's `FaceDetector` for the default framing. Preset portraits
get no such step — they are uploaded as-is — so they must already be right.
