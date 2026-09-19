"""Standalone, local annotation page; source text is rendered with textContent only."""
from pathlib import Path
import hashlib
import json


def make_review(rows,path):
    payload=json.dumps(rows,ensure_ascii=False).replace("<","\\u003c")
    key="analog-review-"+hashlib.sha256(payload.encode()).hexdigest()[:20]
    page='''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Проверка полезности аналогов</title>
<style>
body{font:17px/1.5 system-ui;background:#f2f5fa;color:#17233c;max-width:1080px;margin:32px auto;padding:0 20px}
h1{font-size:30px} .cards{display:grid;grid-template-columns:1fr 1fr;gap:20px}
article,.controls{background:white;border:1px solid #dce2eb;border-radius:14px;padding:22px;margin:16px 0}
.text{white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto}
button,select,textarea{font:inherit;padding:10px;border:1px solid #b4c0d3;border-radius:8px}
button{cursor:pointer;background:#fff;color:#183e81}button.primary{background:#183e81;color:white}
textarea{width:95%;min-height:65px} .muted{color:#5c687c} #warning{color:#a02718}
@media(max-width:700px){.cards{grid-template-columns:1fr}}
</style>
<h1>Полезен ли этот исторический аналог?</h1>
<p class="muted">Оценивайте совпадение проблемы и пользу для следующего шага. Совпадение категории само по себе недостаточно.
Методы поиска и места в выдаче скрыты. Модель решения не генерирует.</p>
<p id="progress"></p><p id="warning"></p>
<div class="cards"><article><h2>Новое обращение</h2><div id="query" class="text"></div></article>
<article><h2>Исторический аналог</h2><p id="record" class="muted"></p><div id="candidate" class="text"></div>
<h3>Результат работ из истории</h3><div id="result" class="text"></div>
<details><summary>Другие записи той же группы</summary><div id="variants" class="text"></div></details></article></div>
<div class="controls"><label>Оценка <select id="grade"><option value="">Не оценено</option>
<option value="0">0 — другая проблема / бесполезно</option><option value="1">1 — похоже, но мало практической пользы</option>
<option value="2">2 — полезно для диагностики или следующего шага</option></select></label>
<p><textarea id="reason" placeholder="Почему? Что совпало или различается?"></textarea></p>
<button id="prev">← Назад</button> <button id="next">Далее →</button>
<button id="unrated">К неоценённому</button> <button class="primary" id="export">Скачать judgments.csv</button>
<p class="muted">Черновик сохраняется в этом браузере, если он разрешает локальное хранение. Регулярно скачивайте CSV.
Не публикуйте страницу: она содержит исходные обращения. Пустую оценку не заменяйте нулём.</p></div>
<script type="application/json" id="data">PAYLOAD</script>
<script>
const rows=JSON.parse(document.getElementById('data').textContent), key='STORAGE_KEY';let i=0, saved={};
const $=id=>document.getElementById(id); const pair=r=>JSON.stringify([r.query_id,r.candidate_group]);
try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){$('warning').textContent='Автосохранение недоступно — скачивайте CSV.'}
function store(){if(!rows.length)return;saved[pair(rows[i])]={relevance:$('grade').value,reason:$('reason').value};
try{localStorage.setItem(key,JSON.stringify(saved))}catch(e){$('warning').textContent='Не удалось сохранить черновик — скачайте CSV.'}}
function render(){if(!rows.length){$('progress').textContent='Пар для оценки нет.';return}const r=rows[i],s=saved[pair(r)]||r;
$('progress').textContent=`Пара ${i+1} из ${rows.length} · Оценено ${rows.filter(r=>(saved[pair(r)]||r).relevance!=='').length}`;
$('query').textContent=r.query_text;$('candidate').textContent=r.candidate_text;$('record').textContent='ID: '+r.candidate_id;
$('result').textContent=r.historical_result||'Решение в истории не указано';
const variants=JSON.parse(r.other_records_json||'[]');$('variants').textContent=variants.length?variants.map(v=>`ID: ${v.id}\n${v.original_text}\nРезультат: ${v.result||'Не указан'}`).join('\\n\\n'):'Нет';
$('grade').value=s.relevance;$('reason').value=s.reason||'';}
$('grade').onchange=()=>{store();render()};$('reason').oninput=store;
$('prev').onclick=()=>{store();i=Math.max(0,i-1);render()};$('next').onclick=()=>{store();i=Math.min(rows.length-1,i+1);render()};
$('unrated').onclick=()=>{store();for(let k=1;k<=rows.length;k++){let j=(i+k)%rows.length;if((saved[pair(rows[j])]||rows[j]).relevance===''){i=j;break}}render()};
$('export').onclick=()=>{store();if(!rows.length)return;const fields=Object.keys(rows[0]),quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';
const lines=[fields,...rows.map(r=>{const s=saved[pair(r)]||r;return fields.map(f=>f==='relevance'?s.relevance:f==='reason'?s.reason:r[f])})];
const blob=new Blob(['\\ufeff'+lines.map(line=>line.map(quote).join(',')).join('\\r\\n')],{type:'text/csv;charset=utf-8'});
const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='judgments.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
render();
</script></html>'''
    Path(path).write_text(page.replace('PAYLOAD',payload).replace('STORAGE_KEY',key),encoding="utf-8")
