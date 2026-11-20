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

/**
 * The LiveKit room of a call: you, the voice agent and the avatar, and what each sends the others.
 *
 * Your microphone reaches the agent; the agent sends its speech to the avatar as byte streams and interrupts it with
 * an RPC; the avatar reports playback back; and the avatar publishes the video and the voice you see and hear, on the
 * agent's behalf. The language model sits outside the room, with the agent.
 */
export function RoomDiagram({ view }: { view: RoomView }) {
  const nodes: Record<string, Node> = {
    you: { x: 70, y: 70, title: "You", detail: "browser", on: view.you },
    agent: { x: 330, y: 70, title: "Voice agent", detail: "LiveKit Agents", on: view.agent },
    avatar: { x: 200, y: 250, title: "Avatar", detail: "Reactor model", on: view.avatar },
  };
  const live = view.connected;
  return (
    <svg className="room-diagram" viewBox="-10 -6 420 346" role="img" aria-label="The LiveKit room of the call">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" className="arrow-head" />
        </marker>
      </defs>

      <rect x="8" y="22" width="384" height="310" rx="16" className={`room-box ${live ? "is-live" : ""}`} />
      <text x="20" y="40" className="room-label">
        LiveKit room
      </text>

      <g className="llm">
        <rect x="290" y="-2" width="104" height="22" rx="11" />
        <text x="342" y="13" textAnchor="middle">
          OpenAI Realtime
        </text>
        <line x1="342" y1="20" x2="342" y2="48" className={`link ${view.agent ? "is-on" : ""}`} />
      </g>

      <Link from={nodes.you} to={nodes.agent} active={view.youSpeaking} dy={-8} label={{ text: "your voice", x: 200, y: 54 }} />
      <Link
        from={nodes.agent}
        to={nodes.avatar}
        active={view.avatarSpeaking}
        dx={10}
        label={{ text: "speech, interrupts", x: 272, y: 158, anchor: "start" }}
      />
      <Link from={nodes.avatar} to={nodes.agent} dx={-10} muted label={{ text: "playback reports", x: 272, y: 172, anchor: "start" }} />
      <Link
        from={nodes.avatar}
        to={nodes.you}
        active={view.avatarSpeaking}
        label={{ text: "video + voice", x: 118, y: 166, anchor: "end" }}
      />

      {Object.entries(nodes).map(([key, n]) => (
        <g key={key} className={`node ${n.on ? "is-on" : ""}`} transform={`translate(${n.x}, ${n.y})`}>
          <rect x="-58" y="-24" width="116" height="48" rx="12" />
          <text y="-3" textAnchor="middle" className="node-title">
            {n.title}
          </text>
          <text y="13" textAnchor="middle" className="node-detail">
            {n.detail}
          </text>
        </g>
      ))}
    </svg>
  );
}

function Link({
  from,
  to,
  label,
  active = false,
  muted = false,
  dx = 0,
  dy = 0,
}: {
  from: Node;
  to: Node;
  label: { text: string; x: number; y: number; anchor?: "start" | "middle" | "end" };
  active?: boolean;
  muted?: boolean;
  dx?: number;
  dy?: number;
}) {
  const len = Math.hypot(to.x - from.x, to.y - from.y);
  const ux = (to.x - from.x) / len;
  const uy = (to.y - from.y) / len;
  // Start and end at the boxes' edges; dx and dy shift a link sideways so a pair of opposite links stays apart.
  const pad = Math.abs(uy) > 0.5 ? 30 : 62;
  return (
    <g className={`link-group ${active ? "is-active" : ""} ${muted ? "is-muted" : ""}`}>
      <line
        x1={from.x + ux * pad + dx}
        y1={from.y + uy * pad + dy}
        x2={to.x - ux * pad + dx}
        y2={to.y - uy * pad + dy}
        className="link"
        markerEnd="url(#arrow)"
      />
      <text x={label.x} y={label.y} textAnchor={label.anchor ?? "middle"} className="link-label">
        {label.text}
      </text>
    </g>
  );
}
