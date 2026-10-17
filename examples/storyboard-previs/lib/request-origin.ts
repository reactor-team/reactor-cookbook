/** Accept the local origin or one explicitly configured HTTPS reverse-proxy origin. */
export function isAllowedOrigin(request: Request, publicOrigin?: string): boolean {
  if (request.headers.get('sec-fetch-site') === 'cross-site') return false;
  const origin = request.headers.get('origin');
  if (!origin || origin === new URL(request.url).origin) return true;
  if (!publicOrigin) return false;
  try {
    const configured = new URL(publicOrigin);
    return configured.protocol === 'https:' && !configured.username && !configured.password
      && configured.pathname === '/' && !configured.search && !configured.hash
      && origin === configured.origin;
  } catch { return false; }
}
