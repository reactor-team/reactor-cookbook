/** A feedback boundary between shots: never prepare N+1 before N finishes playing. */
export async function runReactivePlan<T>(plan: T[], signal: AbortSignal, steps: {
  prepare: (shot: T, index: number) => Promise<T>;
  enqueue: (shot: T, index: number) => Promise<void>;
  wait: (shot: T) => Promise<string>;
  play: (shot: T, clipId: string) => Promise<void>;
}) {
  for (let index = 0; index < plan.length; index++) {
    signal.throwIfAborted();
    const shot = await steps.prepare(plan[index], index);
    signal.throwIfAborted();
    await steps.enqueue(shot, index);
    signal.throwIfAborted();
    const clipId = await steps.wait(shot);
    signal.throwIfAborted();
    await steps.play(shot, clipId);
    signal.throwIfAborted();
  }
}
