import streamlit as st
import hashlib

USERS = {
    "admin": hashlib.sha256("1234".encode()).hexdigest()
}

def login():
    st.sidebar.title("Login")
    user = st.sidebar.text_input("Usuário")
    pwd = st.sidebar.text_input("Senha", type="password")

    if st.sidebar.button("Entrar"):
        if user in USERS and hashlib.sha256(pwd.encode()).hexdigest() == USERS[user]:
            st.session_state["auth"] = True
        else:
            st.sidebar.error("Login inválido")

    return st.session_state.get("auth", False)
