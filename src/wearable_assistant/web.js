'use strict';
let session = null, busy = false, pending = null, snapshot = null;
let polling = null, ticking = null, pollBusy = false, waitStarted = null;
const activePhases = new Set(['running', 'waiting_for_model', 'executing_tool']);
const labels = {ready:'会话已创建',running:'正在处理',waiting_for_model:'模型正在生成',executing_tool:'正在执行工具',
  waiting_for_user:'等待你补充信息',completed:'已完成',failed:'任务失败',partial:'部分完成',cancelled:'已停止后续操作'};
const el = name => document.querySelector('#' + name);

function controls() {
  const active = snapshot && activePhases.has(snapshot.state.phase);
  ['new','send','restore'].forEach(name => { el(name).disabled = busy || Boolean(active); });
  el('retry').disabled = busy || !pending;
  el('stop').disabled = !session;
  el('refresh').disabled = !session;
}

function progress() {
  if (!busy && !(snapshot && activePhases.has(snapshot.state.phase))) { el('progress').textContent = ''; return; }
  const started = snapshot && snapshot.state.phase === 'waiting_for_model' && snapshot.state.model_started_at
    ? snapshot.state.model_started_at * 1000 : waitStarted;
  const seconds = Math.max(0, Math.floor((Date.now() - (started || Date.now())) / 1000));
  const counters = snapshot ? snapshot.counters : {};
  el('progress').textContent = `已等待 ${seconds} 秒；模型请求 ${counters.model_calls || 0} 次，工具调用 ${counters.tool_calls || 0} 次。` +
    '当前为非流式推理，完整回复返回前不会逐字显示；P1000 可能需要数分钟，请勿重复新建。';
}

function show(value) {
  snapshot = value; session = value.session_id;
  try { sessionStorage.setItem('wearable-session', session); } catch (_) {}
  el('sid').textContent = '会话 ' + session;
  el('status').textContent = labels[value.state.phase] || value.state.phase;
  if (busy && value.state.phase === 'ready') el('status').textContent = '请求已提交，正在等待模型…';
  if (value.final && value.final.message && value.state.phase !== 'completed') {
    el('status').textContent += ': ' + value.final.message;
  } else if (value.state.last_error && value.state.phase === 'failed') {
    el('status').textContent += ': ' + (value.state.last_error.includes('truncated') ? '模型输出被截断，请检查模型服务的结束 token 设置。' : value.state.last_error);
  }
  if (value.final_origin === 'controller_weather' && value.state.phase === 'completed') el('status').textContent += '（依据工具结果直接答复）';
  if (value.final_origin === 'controller_query' && value.state.phase === 'completed') el('status').textContent += '（依据查询结果直接答复）';
  if (value.final_origin === 'controller_verified_write' && value.state.phase === 'completed') el('status').textContent += '（系统已核验写入结果）';
  if (value.final_origin === 'controller_event_followup' && value.state.phase === 'completed') el('status').textContent += '（已读取当前活动及关联提醒）';
  if (value.final_origin === 'controller_fast_path' && value.state.phase === 'completed') el('status').textContent += '（已核对工具结果）';
  el('answer').textContent = JSON.stringify({final:value.final,final_origin:value.final_origin,token_usage:value.token_usage,state:value.state,counters:value.counters},null,2);
  el('messages').textContent = '以下 tool 的 success/error 仅表示单次工具状态，整项任务状态以顶部和 final 为准。\n\n' +
    value.messages.map(m => m.role + ': ' + (m.content || JSON.stringify(m.tool_calls))).join('\n\n');
  el('timeline').textContent = value.timeline.map(e => JSON.stringify(e,null,2)).join('\n');
  controls(); progress();
  if (activePhases.has(value.state.phase) || busy) startPolling();
  else stopPolling();
}

async function call(path, body) {
  const headers = {'Content-Type':'application/json'};
  if (el('token').value) headers.Authorization = 'Bearer ' + el('token').value;
  const response = await fetch(path, {method:body ? 'POST' : 'GET', headers,body:body ? JSON.stringify(body) : undefined,cache:'no-store'});
  const value = await response.json();
  if (!response.ok) throw Error(value.error || '请求失败');
  return value;
}

async function poll() {
  if (!session || pollBusy) return;
  const requestedSession = session;
  pollBusy = true;
  try {
    const value = await call('/sessions/' + requestedSession);
    if (session === requestedSession) show(value);
  }
  catch (e) { el('progress').textContent = '状态查询失败: ' + e.message + '。不会自动重新提交任务。'; }
  finally { pollBusy = false; }
}

function startPolling() {
  if (!polling) polling = setInterval(poll, 2000);
  if (!ticking) ticking = setInterval(progress, 1000);
}
function stopPolling() {
  if (polling) clearInterval(polling);
  if (ticking) clearInterval(ticking);
  polling = ticking = null;
}

async function operation(path, body) {
  pending = {path,body:{...body,request_id:crypto.randomUUID()}};
  el('status').textContent = '请求已提交，正在等待模型…';
  startPolling();
  const value = await call(path,pending.body);
  pending = null; show(value);
  return value;
}

async function send(kind) {
  if (busy) return;
  busy = true; waitStarted = Date.now(); controls();
  try {
    if (kind === 'retry') {
      if (!pending) throw Error('没有未确认请求');
      const old = pending;
      show(await call(old.path,old.body)); pending = null;
      if (old.path === '/sessions' && old.body.defer) await operation('/sessions/' + session + '/run',{});
      return;
    }
    if (pending) throw Error('有未确认请求，请先刷新状态或重试原请求');
    const text = el('text').value;
    if (!text.trim()) throw Error('请输入请求');
    if (kind === 'new') {
      session = snapshot = null; stopPolling();
      await operation('/sessions',{request:text,profile:el('profile').value,defer:true});
      await operation('/sessions/' + session + '/run',{});
    } else {
      if (!session) throw Error('请先新建任务');
      await operation('/sessions/' + session + '/messages',{text});
    }
  } catch (e) {
    el('status').textContent = e.message + '；请刷新状态，同一请求重试会复用原 request_id。';
  } finally {
    busy = false; controls(); progress();
    if (!snapshot || !activePhases.has(snapshot.state.phase)) stopPolling();
  }
}

async function stop() {
  if (!session) return;
  try {
    await call('/sessions/' + session + '/cancel',{request_id:crypto.randomUUID()});
    el('status').textContent = '停止请求已记录；当前调用返回后停止，既有写入不会自动撤销。';
  } catch (e) { el('status').textContent = e.message; }
}

async function restore() {
  if (busy) return;
  try {
    const listing = await call('/sessions');
    if (!listing.sessions.length) throw Error('没有已保存会话');
    session = listing.sessions[0].session_id;
    await poll();
  } catch (e) { el('status').textContent = e.message; }
}

el('new').onclick = () => send('new');
el('send').onclick = () => send('send');
el('retry').onclick = () => send('retry');
el('stop').onclick = stop;
el('refresh').onclick = poll;
el('restore').onclick = restore;
controls();
try { session = sessionStorage.getItem('wearable-session'); } catch (_) {}
if (session) poll();
