"""Three-stage review workflow. Never synthesize a different legal game."""
import sys,hashlib,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import streamlit as st
import pandas as pd
import chess,chess.pgn
from ocr_predictor import default_predictor_with_notice,load_rgb_image
from chess_grid import grid_order
from cell_reader import read_slot_images
from move_hints import suggestions

@st.cache_resource
def model():return default_predictor_with_notice()

def validate(rows):
    board=chess.Board();game=chess.pgn.Game();node=game
    moves=[str(r.get(s,'')).strip() for r in rows for s in ('白方','黑方')]
    while moves and not moves[-1]:moves.pop()
    for i,raw in enumerate(moves):
        raw=raw.replace('0','O') if raw in ('0-0','0-0-0','0-0+','0-0-0+') else raw
        try:move=board.parse_san(raw)
        except ValueError:return None,(i,raw)
        node=node.add_variation(move);board.push(move)
    if not moves:return None,(0,'空白')
    return str(game),None

def save_editor(key):
    changes=st.session_state.get(key,{})
    rows=[dict(r) for r in st.session_state['rows']]
    for i,values in changes.get('edited_rows',{}).items():
        for side,value in values.items():
            if side in ('白方','黑方'):rows[int(i)][side]='' if value is None else str(value)
    st.session_state['rows']=rows
    st.session_state.pop('hint_slot',None)
    st.session_state.pop('pgn',None)
    st.session_state['revision']=st.session_state.get('revision',0)+1

def adopt(index,text):
    rows=[dict(r) for r in st.session_state['rows']]
    rows[index//2]['白方' if index%2==0 else '黑方']=text
    st.session_state['rows']=rows
    st.session_state.pop('hint_slot',None)
    st.session_state.pop('pgn',None)
    st.session_state['revision']=st.session_state.get('revision',0)+1
    st.session_state.pop('hint_slot',None)

st.set_page_config(page_title='棋谱转 PGN',layout='wide')
st.title('棋谱转 PGN')
st.caption('上传识别 → 点击候选修正 → 下载 PGN。候选结合字迹与棋规，只修改你选中的一步。')
u=st.file_uploader('上传完整、清晰的棋谱照片',type=['jpg','jpeg','png'])
if u:
    digest=hashlib.sha256(u.getvalue()).hexdigest()
    if st.session_state.get('digest')!=digest:
        st.session_state.clear();st.session_state['digest']=digest
        f=Path(tempfile.gettempdir())/('chess-'+digest+Path(u.name).suffix);f.write_bytes(u.getvalue());st.session_state['path']=str(f)
        st.session_state['pending']=True
    image=load_rgb_image(st.session_state['path'])
    if st.session_state.pop('pending',False):
        try:
            with st.spinner('正在定位表格并识别…'):
                predictor,notice=model()
                raw=predictor.detect_and_recognize(st.session_state['path'],min_score=0,san_like_only=False)
                items,meta=grid_order(image,raw)
                items=read_slot_images(predictor,image,items)
                st.session_state['items']=items;st.session_state['meta']=meta
                st.session_state['rows']=[{'回合':i//2+1,'白方':items[i]['text'],'黑方':items[i+1]['text'] if i+1<len(items) else ''} for i in range(0,len(items),2)]
                st.session_state.pop('error',None)
        except Exception as e:st.session_state['error']=str(e)
    left,right=st.columns([1,1])
    with left:st.image(image,use_column_width=True)
    with right:
        if 'error' in st.session_state:
            st.error('这张照片暂时无法可靠定位回合，已停止自动导出，避免错序。'+st.session_state['error'])
            st.info('可用下方辅助框选固定每一格，避免漏一步后全部错位。框选范围不包含表头。')
            from manual_grid import preview,read_cells
            with st.expander('辅助框选表格',expanded=True):
                xr=st.slider('表格左右边界（百分比）',0,100,(8,95))
                yr=st.slider('第一回合顶部至最后回合底部（百分比）',0,100,(34,96))
                panels=st.selectbox('纸上总栏组数（包括空白栏）',[1,2,3,4],index=2)
                rowcount=st.number_input('每栏回合数',5,60,20)
                grid,box=preview(image,xr,yr,panels,rowcount)
                st.image(grid,caption='红线需与原图格线对齐；倾斜明显时请先拍正照片。',use_column_width=True)
                aligned=st.checkbox('红线已与表格对齐')
                if st.button('按这些格子识别',disabled=not aligned):
                    with st.spinner('正在逐格识别…'):
                        predictor,_=model();items=read_cells(predictor,image,box,panels,rowcount)
                        st.session_state['items']=items
                        st.session_state['rows']=[{'回合':i//2+1,'白方':items[i]['text'],'黑方':items[i+1]['text']} for i in range(0,len(items),2)]
                        st.session_state.pop('error',None);st.rerun()
        if 'rows' in st.session_state:
            rows=st.session_state['rows'];pgn,issue=validate(rows)
            items=st.session_state['items']
            revision=st.session_state.get('revision',0)
            moves=[str(r.get(s,'')).strip() for r in rows for s in ('白方','黑方')]
            st.subheader('电脑提示')
            if issue:
                i,raw=issue;st.warning(f'第 {i//2+1} 回合'+('白方' if i%2==0 else '黑方')+f'无法通过校验：{raw or "空白"}')
            else:st.success('当前走法均合法，仍请核对字迹。')
            uncertain=[i for i,it in enumerate(items) if it.get('needs_review') or it.get('score',0)<.9]
            target=issue[0] if issue else (uncertain[0] if uncertain else 0)
            labels=[f'{j//2+1} 回合'+('白方' if j%2==0 else '黑方')+f' · {moves[j] or "空白"}' for j in range(len(moves))]
            choice=st.selectbox('查看哪一步的字迹和候选（也可检查合法但读错的走法）',labels,index=min(target,len(moves)-1),key='hint_slot')
            selection=labels.index(choice)
            item=items[selection] if selection<len(items) else {}
            box=item.get('cell_bbox',item.get('bbox'))
            if box and box[2]>box[0]:st.image(image.crop(tuple(box)),caption='对应原图',width=260)
            hints,blocked=suggestions(moves,selection,item)
            if blocked is not None:
                st.info(f'请先修正第 {blocked//2+1} 回合'+('白方' if blocked%2==0 else '黑方')+'，才能判断这一步是否合法。')
            elif hints:
                st.caption('以下候选在当前已录入的局面下合法，按字迹概率与文字相似程度排序。前面的合法走法也可能读错。')
                columns=st.columns(len(hints))
                for column,hint in zip(columns,hints):
                    with column:
                        st.button('采用 '+hint['move'],key=f'hint-{selection}-{hint["move"]}',on_click=adopt,args=(selection,hint['move']))
            else:st.info('没有足够的识别信息给出可靠候选，请在下表手动填写。')
            st.caption('修改表格会自动重新校验；点击候选只替换对应的一格。')
            editor_key=f'moves-editor-{revision}'
            st.data_editor(pd.DataFrame(rows),hide_index=True,disabled=['回合'],use_container_width=True,num_rows='fixed',key=editor_key,on_change=save_editor,args=(editor_key,))
            approved=st.checkbox('我已对照原图核对整盘走法',key=f'approved-{revision}')
            if pgn and approved:
                st.download_button('下载 PGN',pgn,file_name='game.pgn',mime='application/x-chess-pgn')
            elif issue:st.caption('修正无法解析的走法后即可导出。')
