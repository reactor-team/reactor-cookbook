// References only guide appearance, so a 768px long edge is plenty for the 1344x768 canvas
// and keeps uploads small on slow venue links.
export async function shrinkImage(blob: Blob, maxEdge = 768, quality = 0.78): Promise<Blob> {
  const bitmap = await createImageBitmap(blob);
  try {
    const scale = Math.min(1, maxEdge / Math.max(bitmap.width, bitmap.height));
    if (scale === 1 && blob.type === 'image/jpeg' && blob.size < 200 * 1024) return blob;
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(bitmap.width * scale); canvas.height = Math.round(bitmap.height * scale);
    const ctx = canvas.getContext('2d'); if (!ctx) return blob;
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const out = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/jpeg', quality));
    return out && out.size < blob.size ? out : blob;
  } finally { bitmap.close(); }
}
