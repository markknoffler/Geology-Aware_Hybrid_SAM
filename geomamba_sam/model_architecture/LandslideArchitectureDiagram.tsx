const C = {
  blue: { f: "#3b82f6", s: "#1d4ed8", t: "#93c5fd" },
  softBlue: "#dbeafe",
  amber: "#fef3c7",
  gray: "#e5e7eb",
  green: "#d1fae5",
  line: "#2563eb",
  bg: "#f8fafc",
};

function Cube({ x, y, w, h, d = 18, label = "" }: any) {
  const sx = d;
  const sy = d * 0.52;
  return (
    <g>
      <polygon points={`${x},${y} ${x + w},${y} ${x + w + sx},${y - sy} ${x + sx},${y - sy}`} fill={C.blue.t} stroke="#1d4ed8" strokeWidth={1} />
      <polygon points={`${x + w},${y} ${x + w + sx},${y - sy} ${x + w + sx},${y + h - sy} ${x + w},${y + h}`} fill={C.blue.s} stroke="#1d4ed8" strokeWidth={1} />
      <rect x={x} y={y} width={w} height={h} fill={C.blue.f} stroke="#1d4ed8" strokeWidth={1.1} rx={2} />
      <text x={x + w / 2} y={y + h + 18} textAnchor="middle" fontSize={13} fill="#1e3a8a" fontStyle="italic">{label}</text>
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, dash = false }: any) {
  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={C.line} strokeWidth={2.2} markerEnd="url(#arr)" strokeDasharray={dash ? "6 5" : undefined} />;
}

function Box({ x, y, w, h, label, fill = "#ffffff" }: any) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={8} fill={fill} stroke="#94a3b8" strokeWidth={1.4} />
      <text x={x + w / 2} y={y + h / 2 + 4} textAnchor="middle" fontSize={12.5} fill="#334155" fontWeight={600}>{label}</text>
    </g>
  );
}

export default function LandslideArchitectureDiagram() {
  const W = 3400;
  const H = 1800;
  return (
    <div style={{ background: "#ffffff", minHeight: "100vh", padding: 10 }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: "100%", height: "auto", background: "#ffffff" }}>
        <defs>
          <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3.5" orient="auto">
            <polygon points="0 0,8 3.5,0 7" fill="#2563eb" />
          </marker>
        </defs>

        <rect x={20} y={20} width={3360} height={1680} rx={80} fill={C.bg} stroke="#93c5fd" strokeWidth={4} strokeDasharray="9 7" />

        {/* Left major branch */}
        <rect x={120} y={100} width={1810} height={1500} rx={28} fill="#ffffff" stroke="#cbd5e1" strokeWidth={2.2} />
        <text x={180} y={160} fontSize={34} fontWeight={700} fill="#0f172a">Input Image</text>
        <text x={1700} y={160} fontSize={42} fill="#1e3a8a" fontStyle="italic">Encoder Path</text>
        <text x={1700} y={1530} fontSize={42} fill="#b45309" fontStyle="italic">Decoder Path</text>

        {/* pseudo image tiles */}
        <rect x={170} y={220} width={210} height={170} fill="#f59e0b" stroke="#111827" strokeWidth={2} />
        <rect x={170} y={450} width={210} height={140} fill="#111827" stroke="#111827" strokeWidth={2} />
        <circle cx={220} cy={500} r={16} fill="#22c55e" /><circle cx={282} cy={542} r={13} fill="#22c55e" /><circle cx={260} cy={486} r={10} fill="#22c55e" />
        <rect x={170} y={660} width={210} height={140} fill="#111827" stroke="#111827" strokeWidth={2} />
        <circle cx={238} cy={706} r={13} fill="#22c55e" /><circle cx={300} cy={744} r={12} fill="#22c55e" />
        <text x={405} y={525} fontSize={28} fill="#64748b" fontStyle="italic">GT</text>
        <text x={405} y={740} fontSize={28} fill="#64748b" fontStyle="italic">Pred</text>
        <text x={360} y={620} fontSize={28} fill="#64748b" fontStyle="italic">supervision</text>
        <Arrow x1={385} y1={305} x2={570} y2={305} />

        {/* encoder body */}
        <rect x={560} y={130} width={1300} height={540} rx={40} fill={C.amber} stroke="#cbd5e1" strokeWidth={2.2} />
        <text x={650} y={130} fontSize={30} fill="#374151" fontStyle="italic">f1: H/2xW/2xC1</text>
        <text x={940} y={130} fontSize={30} fill="#374151" fontStyle="italic">f2: H/4xW/4xC2</text>
        <text x={1225} y={130} fontSize={30} fill="#374151" fontStyle="italic">f3: H/8xW/8xC3</text>
        <text x={1505} y={130} fontSize={30} fill="#374151" fontStyle="italic">f4: H/16xW/16xC4</text>

        <Cube x={720} y={250} w={90} h={270} label="" />
        <Cube x={1040} y={250} w={78} h={250} label="" />
        <Cube x={1335} y={275} w={68} h={210} label="" />
        <Cube x={1600} y={315} w={74} h={150} label="" />
        <Arrow x1={825} y1={330} x2={1028} y2={330} />
        <Arrow x1={1135} y1={330} x2={1325} y2={335} />
        <Arrow x1={1418} y1={335} x2={1590} y2={350} />

        {/* mid strip */}
        <rect x={560} y={700} width={1300} height={370} rx={34} fill={C.gray} stroke="#cbd5e1" strokeWidth={2.2} />
        {[
          [730, "PFF1", "DAB1"],
          [1025, "PFF2", "DAB2"],
          [1320, "PFF3", "DAB3"],
          [1615, "PFF4", "DAB4"],
        ].map((v: any, i: number) => (
          <g key={i}>
            <Box x={v[0]} y={760} w={120} h={56} label={v[1]} />
            <Arrow x1={v[0] + 60} y1={520} x2={v[0] + 60} y2={756} />
            <Box x={v[0]} y={870} w={120} h={56} label={v[2]} fill="#f0f9ff" />
            <Arrow x1={v[0] + 60} y1={816} x2={v[0] + 60} y2={868} />
          </g>
        ))}
        <Arrow x1={765} y1={520} x2={1085} y2={760} />
        <Arrow x1={1078} y1={500} x2={1380} y2={760} />
        <Arrow x1={1370} y1={500} x2={1680} y2={760} />
        <Arrow x1={1668} y1={465} x2={1680} y2={760} />

        {/* decoder strip */}
        <rect x={560} y={1130} width={1300} height={120} rx={22} fill="#fdba74" stroke="#cbd5e1" strokeWidth={2.2} />
        {[
          [640, "UP, Conv"],
          [930, "UP, Conv"],
          [1220, "UP, Conv"],
          [1510, "UP, Conv"],
        ].map((v: any, i: number) => (
          <g key={i}>
            <Box x={v[0]} y={1158} w={160} h={64} label={v[1]} fill="#fff7ed" />
            {i < 3 ? <Arrow x1={v[0] + 165} y1={1190} x2={v[0] + 275} y2={1190} /> : null}
            {i > 0 ? <text x={v[0] - 16} y={1202} fontSize={50} fill="#334155">⊕</text> : null}
          </g>
        ))}
        <Arrow x1={790} y1={928} x2={790} y2={1158} />
        <Arrow x1={1085} y1={928} x2={1085} y2={1158} />
        <Arrow x1={1380} y1={928} x2={1380} y2={1158} />
        <Arrow x1={1675} y1={928} x2={1675} y2={1158} />

        {/* Right upper: progressive fusion */}
        <rect x={1980} y={130} width={1280} height={540} rx={60} fill="#e5e7eb" stroke="#6b7280" strokeWidth={3} />
        <text x={2410} y={620} fontSize={48} fontStyle="italic" fill="#374151">Progressive Feature Fusion</text>
        <text x={2010} y={240} fontSize={52} fill="#374151" fontStyle="italic">f(i+1)</text>
        <text x={2010} y={355} fontSize={52} fill="#374151" fontStyle="italic">f(i)</text>
        <text x={2010} y={470} fontSize={52} fill="#374151" fontStyle="italic">f(i-1)</text>
        <Box x={2240} y={190} w={180} h={72} label="Conv1x1" />
        <Box x={2460} y={190} w={120} h={72} label="UP" />
        <Box x={2240} y={410} w={180} h={72} label="DWConv" />
        <Box x={2460} y={410} w={180} h={72} label="Conv1x1" />
        <Arrow x1={2100} y1={230} x2={2235} y2={230} />
        <Arrow x1={2585} y1={230} x2={2800} y2={230} />
        <Arrow x1={2100} y1={355} x2={2800} y2={355} />
        <Arrow x1={2100} y1={450} x2={2235} y2={450} />
        <Arrow x1={2645} y1={450} x2={2800} y2={450} />
        <text x={2800} y={245} fontSize={56} fill="#334155">⊗</text>
        <text x={2800} y={370} fontSize={56} fill="#334155">⊗</text>
        <text x={2890} y={315} fontSize={56} fill="#334155">⊙</text>
        <Cube x={3010} y={240} w={130} h={190} label="f(i)PFF" />
        <Arrow x1={2920} y1={315} x2={3008} y2={315} />
        <text x={2865} y={250} fontSize={46} fill="#374151" fontStyle="italic">f''(i+1)</text>
        <text x={2865} y={430} fontSize={46} fill="#374151" fontStyle="italic">f''(i-1)</text>

        {/* Right lower: dynamic attention */}
        <rect x={1980} y={730} width={1280} height={740} rx={60} fill="#dbeafe" stroke="#93c5fd" strokeWidth={3} />
        <text x={2390} y={1430} fontSize={50} fontStyle="italic" fill="#374151">Dynamic Attention Block</text>
        <Cube x={2050} y={1000} w={120} h={210} label="" />
        {[
          [2240, 900, "Avg.FC"],
          [2435, 900, "Sigmoid"],
          [2240, 1060, "Cavg"],
          [2435, 1060, "Sigmoid"],
          [2240, 1220, "Conv1x1"],
          [2435, 1220, "Sigmoid"],
        ].map((b: any, i: number) => (
          <Box key={i} x={b[0]} y={b[1]} w={170} h={68} label={b[2]} />
        ))}
        <Arrow x1={2175} y1={1030} x2={2235} y2={934} />
        <Arrow x1={2175} y1={1090} x2={2235} y2={1094} />
        <Arrow x1={2175} y1={1160} x2={2235} y2={1254} />
        <Arrow x1={2610} y1={934} x2={2865} y2={934} />
        <Arrow x1={2610} y1={1094} x2={2865} y2={1094} />
        <Arrow x1={2610} y1={1254} x2={2865} y2={1254} />
        <text x={2865} y={950} fontSize={50} fill="#334155">⊗</text>
        <text x={2865} y={1110} fontSize={50} fill="#334155">⊗</text>
        <text x={2865} y={1270} fontSize={50} fill="#334155">⊗</text>
        <Cube x={2960} y={860} w={120} h={130} label="" />
        <Cube x={2960} y={1020} w={120} h={130} label="" />
        <Cube x={2960} y={1180} w={120} h={130} label="" />
        <Arrow x1={2915} y1={934} x2={2958} y2={934} />
        <Arrow x1={2915} y1={1094} x2={2958} y2={1094} />
        <Arrow x1={2915} y1={1254} x2={2958} y2={1254} />
        <Arrow x1={3090} y1={934} x2={3195} y2={934} />
        <Arrow x1={3090} y1={1094} x2={3195} y2={1094} />
        <Arrow x1={3090} y1={1254} x2={3195} y2={1254} />
        <text x={3195} y={1110} fontSize={52} fill="#334155">⊕</text>

        {/* captions */}
        <text x={118} y={1660} fontSize={26} fill="#334155">Two-stage architecture: classifier-gated GSAM segmentation with progressive feature fusion and dynamic attention.</text>
      </svg>
    </div>
  );
}

