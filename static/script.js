let currentSession = localStorage.getItem("currentSession");

const chatContainer = document.getElementById("chatContainer");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");

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

        item.textContent =
            session.title;

        item.dataset.id =
            session.id;

        item.addEventListener(
            "click",
            () => {

                currentSession =
                    session.id;

                localStorage.setItem(
                    "currentSession",
                    currentSession
                );
            }
        );

        list.appendChild(item);
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

wrapper.innerHTML = `
    <div>${text}</div>
    <small>${new Date().toLocaleTimeString()}</small>
`;

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

window.onload = async () => {

if (!currentSession) {
    await createSession();
}

await loadSessions();

};