// Results page: data/sg.json (site_hooks.extra) as tables.
const ci = (c) => c && c[0] != null ? `${pct(c[0])} – ${pct(c[1])}` : '–';
const tbl = (head, rows) => `<div class="tablewrap"><table><tr>${head.map((h) => `<th>${esc(h)}</th>`).join('')}</tr>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')}</table></div>`;
get('sg.json').then((d) => {
  if (!d) return;
  const A = (d.ablation || {}).arms || {}, live = d.ablation_live || {};
  const arms = ['mirror', 'buy', 'action', 'both'];
  $('abl').innerHTML = tbl(['arm', 'kingdom pairs', 'score', '95% interval', 'buy overrides', 'action overrides', 'capped / errored games', 'latency p50 / p95 (s)', 'rollouts per game'],
    arms.map((a) => { const s = A[a]; const l = live[a] || {};
      return s ? [a, s.pairs, pct(s.score), ci(s.ci95), pct(s.buy_override_rate, 2), pct(s.action_override_rate, 2), `${s.capped_games} / ${s.errored_games}`,
        s.latency_s ? `${s.latency_s.p50} / ${s.latency_s.p95}` : '–', s.rollouts_per_game]
        : [a, l.pairs ? `${l.pairs} (playing)` : 'waiting', l.score != null ? pct(l.score) + ' so far' : '–', '–', '–', '–', '–', '–', '–']; }));
  const P = (d.ablation || {}).paired || {};
  $('ablp').innerHTML = Object.keys(P).length ? '<h3>Paired differences on the same games</h3>' + tbl(['comparison', 'pairs', 'difference (points)', '95% interval'],
    Object.entries(P).map(([k, v]) => [esc(k), v.pairs, (100 * v.diff).toFixed(1), `${(100 * v.ci95[0]).toFixed(1)} – ${(100 * v.ci95[1]).toFixed(1)}`])) : '';
  let h = '';
  if (d.pilot) h += `<p>Pilot run <code>${esc(d.pilot.run)}</code>, cohort <code>${esc(d.pilot.cohort)}</code>, ${d.pilot.shards} shards, labels: ${esc(d.pilot.kinds)}.</p>`;
  if (d.bank && d.bank.all) { const b = d.bank.all;
    h += '<h3>Held-out bank</h3>' + tbl(['positions', 'c2 regret', 'shortlist regret', 'best option outside the shortlist', 'teacher overrides', 'their mean gain', 'harmful / good'],
      [[d.bank.positions, b.incumbent_regret, b.shortlist_regret, pct(b.strict_outside_shortlist), pct(b.teacher_override_rate), b.teacher_mean_gain, `${pct(b.teacher_harmful)} / ${pct(b.teacher_good)}`]]); }
  if (d.survive) h += '<h3>Do the proposed overrides survive?</h3>' + tbl(['view', 'overrides', 'still > margin', 'still positive', 'mean gain'],
    Object.entries(d.survive.views).map(([k, v]) => [esc(k), v.n, pct(v.survive), pct(v.positive), v.mean_gain]));
  $('pilot').innerHTML = h || '<p class="muted">After the ablation.</p>';
  const S = d.supervise || {};
  $('sup').innerHTML = Object.keys(S).length ? tbl(['run', 'verified pairs (train / dev / test)', 'held-out pair accuracy before → after', 'takes the winner after', 'anchor KL', 'anchor argmax agreement', 'value drift'],
    Object.entries(S).map(([k, s]) => [esc(k), `${s.verified.train} / ${s.verified.dev} / ${s.verified.test}`,
      s.first.test ? `${pct(s.first.test.pair_acc)} → ${pct(s.last.test.pair_acc)}` : '–', s.last.test ? pct(s.last.test.takes_winner) : '–',
      s.last.anchor_test.all.kl, pct(s.last.anchor_test.all.argmax_agree, 2), s.last.anchor_test.value_abs_drift])) : '<p class="muted">After the pilot.</p>';
  const R = d.ppo || {};
  $('ppo').innerHTML = Object.keys(R).length ? tbl(['run', 'iteration', 'vs frozen c2 (eval)', 'vs frozen c1 (eval)', 'vs GA base (eval)', 'training games', 'CPU hours'],
    Object.entries(R).map(([k, r]) => [esc(k), r.iter, pct(+r.eval_vs_c2), pct(+r.eval_vs_c1), pct(+r.eval_GA_base), fmtM(r.games_total), (+r.cpu_hours || 0).toFixed(1)])) +
    '<p class="muted small">The per-iteration evaluation uses 200 games per opponent: a progress signal, not a test. See the Training tab for the curves.</p>' : '<p class="muted">After the supervised step.</p>';
  const M = d.matches || {};
  $('matches').innerHTML = Object.keys(M).length ? tbl(['match', 'pairs', 'score', '95% interval', 'capped', 'errors'],
    Object.entries(M).map(([k, m]) => [esc(k), m.pairs, pct(m.score), ci(m.ci95), m.capped, m.errors])) : '<p class="muted">None yet.</p>';
  if (d.decision) $('decision').innerHTML = `<p>${esc(d.decision.summary || '')}</p>`;
});
