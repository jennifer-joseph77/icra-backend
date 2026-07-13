let currentSession = localStorage.getItem("currentSession");

const chatContainer = document.getElementById("chatContainer");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");

DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A") {
        node.setAttribute("target", "_blank");
        node.setAttribute("rel", "noopener noreferrer");
    }
});

async function createSession() {
const response = await fetch("/sessions", {
method: "POST"
});

const data = await response.json();

currentSession = data.session_id;

localStorage.setItem(
    "currentSession",
    currentSession
);

await loadSessions();


}

async function loadSessions() {

    try {

        const response =
            await fetch("/sessions");

        const sessions =
            await response.json();

        const list =
            document.getElementById("chat-list");

        if (!list) return;

        list.innerHTML = "";

        sessions.forEach(session => {

            const item =
                document.createElement("div");

            item.className =
                "chat-item";
                if (session.id === currentSession) {
                    item.classList.add("active-chat");
                }

            const titleSpan = document.createElement("span");
            titleSpan.textContent = session.title;

            const deleteSpan = document.createElement("span");
            deleteSpan.className = "delete-chat";
            deleteSpan.textContent = "🗑";

            item.appendChild(titleSpan);
            item.appendChild(deleteSpan);

            item.addEventListener(
                "click",
                async () => {

                    currentSession = session.id;

                    localStorage.setItem(
                        "currentSession",
                        currentSession
                    );

                    await loadSessions(); // refresh active highlight

                    await loadConversation(
                        session.id
                    );
                }
            );

            const deleteBtn = deleteSpan;

            deleteBtn.addEventListener(
                "click",
                async (e) => {

                    e.stopPropagation();

                    const confirmDelete =
                        confirm(
                            "Delete this chat?"
                        );

                    if (!confirmDelete)
                        return;

                    await fetch(
                        `/sessions/${session.id}`,
                        {
                            method: "DELETE"
                        }
                    );

                    await loadSessions();

                    if (currentSession === session.id) {

                        currentSession = null;

                        localStorage.removeItem(
                            "currentSession"
                        );

                        await createSession();

                        chatContainer.innerHTML = `
                            <div class="welcome">
                                <h1>ICRA</h1>
                                <p>How can I help you today?</p>
                            </div>
                        `;
                    }
                }
            );

            list.appendChild(item);
        });

    } catch (err) {

        console.error(err);
    }
}

async function loadConversation(
    sessionId
) {

    try {

        const response =
            await fetch(
                `/sessions/${sessionId}/messages`
            );

        const messages =
            await response.json();

        chatContainer.innerHTML = "";

        messages.forEach(msg => {

            addMessage(
                msg.content,
                msg.role === "assistant"
                    ? "bot"
                    : "user"
            );

        });

    } catch (err) {

        console.error(err);
    }
}

function addMessage(text, role) {


if (!chatContainer) return;

const wrapper =
    document.createElement("div");

wrapper.className =
    "message " + role;

const textDiv = document.createElement("div");
textDiv.className = "message-content";

if (role === "bot") {
    textDiv.innerHTML = DOMPurify.sanitize(marked.parse(text));
} else {
    textDiv.textContent = text;
}

const timeSmall = document.createElement("small");
timeSmall.textContent = new Date().toLocaleTimeString();

wrapper.appendChild(textDiv);
wrapper.appendChild(timeSmall);

chatContainer.appendChild(wrapper);

chatContainer.scrollTop =
    chatContainer.scrollHeight;


}

async function askQuestion(question) {

try {

    const response =
        await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type":
                "application/json"
            },
            body: JSON.stringify({
                question: question,
                session_id: currentSession
        })
        });

    const data =
        await response.json();

    addMessage(
        data.answer,
        "bot"
    );
await loadSessions();
} catch (err) {

    addMessage(
        "Server error.",
        "bot"
    );

    console.error(err);
}


}

async function startNewChat() {

await createSession();

chatContainer.innerHTML = `
    <div class="welcome">
        <h1>ICRA</h1>
        <p>How can I help you today?</p>
    </div>
`;

}


chatForm.addEventListener(
"submit",
async (e) => {

    e.preventDefault();

    const question =
        questionInput.value.trim();

    if (!question) return;

    addMessage(
        question,
        "user"
    );

    questionInput.value = "";

    await askQuestion(
        question
    );
}

);

function toggleTheme() {

    document.body.classList.toggle(
        "light-mode"
    );

    const btn =
        document.querySelector(
            ".theme-toggle"
        );

    if (
        document.body.classList.contains(
            "light-mode"
        )
    ) {

        btn.innerHTML =
            "☀️ Light Mode";

        localStorage.setItem(
            "theme",
            "light"
        );

    } else {

        btn.innerHTML =
            "🌙 Dark Mode";

        localStorage.setItem(
            "theme",
            "dark"
        );
    }
}

window.onload = async () => {

    const savedTheme =
        localStorage.getItem(
            "theme"
        );

    if (savedTheme === "light") {

        document.body.classList.add(
            "light-mode"
        );

        const btn =
            document.querySelector(
                ".theme-toggle"
            );

        if (btn) {
            btn.innerHTML =
                "☀️ Light Mode";
        }
    }

    if (!currentSession) {
        await createSession();
    }

    await loadSessions();

    if (currentSession) {
        await loadConversation(
            currentSession
        );
    }
};
