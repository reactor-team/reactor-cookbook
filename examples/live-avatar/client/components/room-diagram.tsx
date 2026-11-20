"use client";

/** Who is in the room and what is flowing right now; drives the highlights. */
export interface RoomView {
  connected: boolean;
  you: boolean;
  agent: boolean;
  avatar: boolean;
  youSpeaking: boolean;
  avatarSpeaking: boolean;
}

interface Node {
  x: number;
  y: number;
  title: string;
  detail: string;
  on: boolean;
}

const W = 112;
const H = 46;

/**
 * The LiveKit room of a call: you, the voice agent and the avatar, and what each sends the others.
 *
 * Your microphone reaches the agent; the agent sends its speech to the avatar as byte streams and interrupts it with
 * an RPC; the avatar reports playback back; and the avatar publishes the video and the voice you see and hear, on the
 * agent's behalf. The language model sits outside the room, with the agent.
 */
export function RoomDiagram({ view }: { view: RoomView }) {
  const you: Node = { x: 66, y: 60, title: "You", detail: "browser", on: view.you };
  const agent: Node = { x: 294, y: 60, title: "Agent", detail: "voice agent", on: view.agent };
  const avatar: Node = { x: 180, y: 214, title: "Avatar", detail: "Reactor", on: view.avatar };
  return (
    <svg className="room-diagram" viewBox="0 0 360 262" role="img" aria-label="The LiveKit room of the call">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" className="arrow-head" />
        </marker>
      </defs>

      <Line x1={you.x + W / 2} y1={you.y} x2={agent.x - W / 2} y2={agent.y} active={view.youSpeaking} />
      <Label x={180} y={you.y - 8} text="your voice" active={view.youSpeaking} />

      <Line x1={agent.x - 22} y1={agent.y + H / 2} x2={avatar.x + 30} y2={avatar.y - H / 2} active={view.avatarSpeaking} />
      <Line x1={avatar.x + 48} y1={avatar.y - H / 2} x2={agent.x - 4} y2={agent.y + H / 2} muted />
      <Label x={272} y={142} text="speech" active={view.avatarSpeaking} anchor="start" />
      <Label x={272} y={158} text="playback" anchor="start" />

      <Line x1={avatar.x - 30} y1={avatar.y - H / 2} x2={you.x + 10} y2={you.y + H / 2} active={view.avatarSpeaking} />
      <Label x={88} y={150} text="video + voice" active={view.avatarSpeaking} anchor="end" />

      {[you, agent, avatar].map((n) => (
        <g key={n.title} className={`node ${n.on ? "is-on" : ""}`} transform={`translate(${n.x}, ${n.y})`}>
          <rect x={-W / 2} y={-H / 2} width={W} height={H} rx={12} />
          <text y={-2} textAnchor="middle" className="node-title">
            {n.title}
          </text>
          <text y={14} textAnchor="middle" className="node-detail">
            {n.detail}
          </text>
        </g>
      ))}
    </svg>
  );
}

function Line({
  x1,
  y1,
  x2,
  y2,
  active = false,
  muted = false,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  active?: boolean;
  muted?: boolean;
}) {
  return (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      className={`link ${active ? "is-active" : ""} ${muted ? "is-muted" : ""}`}
      markerEnd="url(#arrow)"
    />
  );
}

function Label({
  x,
  y,
  text,
  active = false,
  anchor = "middle",
}: {
  x: number;
  y: number;
  text: string;
  active?: boolean;
  anchor?: "start" | "middle" | "end";
}) {
  return (
    <text x={x} y={y} textAnchor={anchor} className={`link-label ${active ? "is-active" : ""}`}>
      {text}
    </text>
  );
}
