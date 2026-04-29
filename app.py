import streamlit as st
import numpy as np
import gymnasium as gym
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

st.set_page_config(page_title="Taxi RL", layout="wide")

if "env" not in st.session_state:
    st.session_state.update({"step":0,"done":False,"rl":0,"rt":0,"history":[]})
    st.session_state.env = gym.make("Taxi-v3")
    st.session_state.state, _ = st.session_state.env.reset()

Q = np.load("q_table.npy")
LOCS   = [(0,0),(0,4),(4,0),(4,3)]
COLORS = ["#c0392b","#27ae60","#b7950b","#2471a3"]
LABELS = "RGYB"
WALLS  = {(0,1),(1,1),(3,0),(4,0),(3,2),(4,2)}

def decode(s):
    return s//20//5, s//20%5, (s//4)%5, s%4

def render(state):
    row, col, pas, dst = decode(state)
    cells = ""
    for r in range(5):
        for c in range(5):
            bg = "#3a3a3a"
            inner = ""
            for i,(lr,lc) in enumerate(LOCS):
                if (r,c)==(lr,lc):
                    bg = COLORS[i]
                    inner += f'<b class="lbl">{LABELS[i]}</b>'
            if (r,c)==LOCS[dst]:
                inner += '<span class="house">🏠</span>'
            if pas<4 and (r,c)==LOCS[pas]:
                inner += '<span class="person">🧍</span>'
            if (r,c)==(row,col):
                inner += '<span class="taxi">🚕</span>'
                if pas==4:
                    inner += '<span class="person_in">🧍</span>'
            wall = "wall" if (r,c) in WALLS else ""
            cells += f'<div class="cell {wall}" style="background:{bg}">{inner}</div>'

    rc = "#e74c3c" if st.session_state.rl < 0 else "#2ecc71"
    return f"""
<style>
body{{background:transparent;margin:0;padding:0}}
.grid{{
    display:grid;
    grid-template-columns:repeat(5,80px);
    gap:3px;
    width:fit-content;
    margin:0 auto;
    padding:8px;
    background:#1a1a1a;
    border-radius:10px;
    border:2px solid #444;
}}
.cell{{width:80px;height:80px;border-radius:6px;position:relative;
    display:flex;align-items:center;justify-content:center}}
.cell.wall{{border-right:4px solid #111}}
.lbl{{position:absolute;top:3px;left:5px;font-size:11px;color:rgba(255,255,255,0.7)}}
.house{{position:absolute;bottom:3px;right:4px;font-size:18px}}
.person{{position:absolute;font-size:22px}}
.taxi{{position:absolute;font-size:28px;filter:drop-shadow(0 0 4px #f1c40f)}}
.person_in{{position:absolute;top:3px;right:3px;font-size:13px}}
</style>
<div class="grid">{cells}</div>
"""

def do_step():
    a = int(np.argmax(Q[st.session_state.state]))
    ns, r, t1, t2, _ = st.session_state.env.step(a)
    st.session_state.history.append(int(r))
    st.session_state.update({"state":ns,"done":t1 or t2,
                              "rl":int(r),"rt":st.session_state.rt+int(r)})
    st.session_state.step += 1

# --- UI ---
st.title("🚕 Taxi RL (Q-learning)")
col1, col2 = st.columns([3, 2])

with col1:
    b1, b2, b3 = st.columns(3)
    if b1.button("▶️ Step"):
        if not st.session_state.done: do_step(); st.rerun()
        else: st.warning("Игра закончена → сбрось")
    if b2.button("⏭️ Auto"):
        while not st.session_state.done: do_step()
        st.rerun()
    if b3.button("🔄 Reset"):
        st.session_state.update({"step":0,"done":False,"rl":0,"rt":0,"history":[]})
        st.session_state.env = gym.make("Taxi-v3")
        st.session_state.state, _ = st.session_state.env.reset()
        st.rerun()

    components.html(render(st.session_state.state), height=460)

    if st.session_state.done:
        st.balloons()
        st.success(f"🎉 Готово! Итог: {st.session_state.rt}")

with col2:
    st.subheader("📊 Статистика")
    m1, m2, m3 = st.columns(3)
    m1.metric("Шаги",  st.session_state.step)
    m2.metric("Награда", f"{st.session_state.rl:+d}")
    m3.metric("Итого", f"{st.session_state.rt:+d}")

    st.markdown("---")
    st.markdown("**Легенда:**")
    st.markdown("🚕 Такси &nbsp;&nbsp; 🧍 Пассажир &nbsp;&nbsp; 🏠 Цель")
    st.markdown("🔴 R &nbsp; 🟢 G &nbsp; 🟡 Y &nbsp; 🔵 B — остановки")

    st.markdown("---")
    st.subheader("📈 Награды")
    if st.session_state.history:
        st.line_chart(st.session_state.history)
    else:
        st.info("Нажми Step или Auto")

    st.markdown("---")
    st.subheader("📈 График обучения")
    
    try:
        rewards = np.load("train_rewards.npy")
        window = 100
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(smoothed, color='green', linewidth=2)
            ax.set_xlabel("Эпизоды")
            ax.set_ylabel("Награда")
            ax.set_title("Training Progress")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info(f"Данных мало: {len(rewards)} эпизодов")
    except:
        st.warning("Нет файла train_rewards.npy. Запусти train.py")