// Real shipped JS in a small DOM/fetch fixture: no browser or model request.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

function state(phase='ready') {
  return {session_id:'fixture-session',profile:'full',state:{phase,model_started_at:Date.now()/1000-8,last_error:null},
    counters:{model_calls:phase==='ready'?0:1,tool_calls:0},messages:[],timeline:[],final:null};
}
function page(fetcher,stored=null) {
  const elements={};let sequence=0;const timers=new Map();let timerId=0;
  for(const name of ['token','profile','text','status','progress','sid','answer','messages','timeline','new','send','stop','refresh','retry','restore'])
    elements[name]={value:name==='profile'?'full':name==='text'?'查询杭州明天的天气。':'',disabled:false,textContent:''};
  const ctx={document:{querySelector:selector=>elements[selector.slice(1)]},
    sessionStorage:{getItem:()=>stored,setItem:(_,v)=>{stored=v;}},
    crypto:{randomUUID:()=>`request-${++sequence}`},Date,JSON,Set,Boolean,Error,console,
    setInterval:fn=>{timers.set(++timerId,fn);return timerId;},clearInterval:id=>timers.delete(id),fetch:fetcher};
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync('src/wearable_assistant/web.js','utf8'),ctx);
  return {ctx,elements,timers};
}
const response=value=>({ok:true,json:async()=>value});
const flush=()=>new Promise(resolve=>setImmediate(resolve));

test('new task immediately shows waiting, polls progress and never duplicates generation',async()=>{
  let finish;const calls=[];
  const {ctx,elements}=page(async(path,options)=>{
    calls.push({path,body:options.body&&JSON.parse(options.body)});
    if(path==='/sessions')return response(state());
    if(path.endsWith('/run'))return await new Promise(resolve=>{finish=resolve;});
    return response(state('waiting_for_model'));
  });
  const running=ctx.send('new');await flush();
  assert.match(elements.status.textContent,/等待模型/);
  assert.equal(elements.new.disabled,true);
  assert.equal(elements.refresh.disabled,false);
  assert.equal(elements.stop.disabled,false);
  await ctx.poll();
  assert.match(elements.status.textContent,/正在生成/);
  assert.match(elements.progress.textContent,/已等待 \d+ 秒/);
  assert.match(elements.progress.textContent,/模型请求 1 次/);
  const failed=state('failed');failed.state.last_error='Model response was truncated';failed.final={status:'failed'};
  finish(response(failed));await running;
  assert.match(elements.status.textContent,/截断/);
  assert.equal(elements.new.disabled,false);
  assert.equal(calls.filter(c=>c.path.endsWith('/run')).length,1);
});

test('restoring saved session only reads progress; it does not resubmit',async()=>{
  const calls=[];
  const {elements}=page(async(path,options)=>{calls.push([path,options.method]);return response(state('failed'));},'fixture-session');
  await flush();
  assert.deepEqual(calls,[['/sessions/fixture-session','GET']]);
  assert.match(elements.status.textContent,/失败/);
});

test('cancel remains available while model request is pending',async()=>{
  let finish;const calls=[];
  const {ctx}=page(async(path,options)=>{
    calls.push(path);
    if(path==='/sessions')return response(state());
    if(path.endsWith('/run'))return await new Promise(resolve=>{finish=resolve;});
    return response({cancellation_requested:true});
  });
  const running=ctx.send('new');await flush();await ctx.stop();
  assert.equal(calls.filter(c=>c.endsWith('/cancel')).length,1);
  finish(response(state('cancelled')));await running;
});

test('network retry reuses the original request_id',async()=>{
  let failures=0;const runIds=[];
  const {ctx}=page(async(path,options)=>{
    if(path==='/sessions')return response(state());
    if(path.endsWith('/run')){
      runIds.push(JSON.parse(options.body).request_id);
      if(!failures++)throw Error('network fixture');
      return response(state('completed'));
    }
    return response(state());
  });
  await ctx.send('new');await ctx.send('retry');
  assert.equal(runIds.length,2);assert.equal(runIds[0],runIds[1]);
});

test('task budget failure overrides a successful last tool receipt',()=>{
  const {ctx,elements}=page(async()=>response(state()));
  const value=state('failed');
  value.state.last_error='NOT_FOUND';
  value.final={status:'failed',message:'已达到服务端报告的累计 token 预算。'};
  value.messages=[{role:'tool',content:'{"status":"success"}'}];
  value.token_usage={reported_total_tokens:32800,max_total_tokens:32768};
  ctx.show(value);
  assert.match(elements.status.textContent,/任务失败.*累计 token 预算/);
  assert.doesNotMatch(elements.status.textContent,/NOT_FOUND/);
  assert.match(elements.messages.textContent,/单次工具状态/);
  assert.match(elements.answer.textContent,/32800/);
});

test('controller answer is labelled separately from a model completion',()=>{
  const {ctx,elements}=page(async()=>response(state()));
  const value=state('completed');value.final_origin='controller_weather';
  value.final={status:'success',message:'本地模拟天气：杭州 2026-11-19 有雨。'};
  ctx.show(value);
  assert.match(elements.status.textContent,/已完成.*工具结果直接答复/);
  assert.match(elements.answer.textContent,/controller_weather/);
});

test('query and verified write completions disclose controller assistance',()=>{
  const {ctx,elements}=page(async()=>response(state()));
  for(const [origin,label] of [['controller_query',/查询结果直接答复/],['controller_verified_write',/系统已核验写入结果/]]){
    const value=state('completed');value.final_origin=origin;
    value.final={status:'success',message:'fixture'};ctx.show(value);
    assert.match(elements.status.textContent,label);
    assert.ok(elements.answer.textContent.includes(origin));
  }
});
