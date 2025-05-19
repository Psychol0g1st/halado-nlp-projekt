"use client";

import { useRef, useState, useEffect } from "react";
import styles from "./Chat.module.css";
import { PaperPlaneIcon } from "@radix-ui/react-icons";

interface Message {
  sender: "user" | "bot";
  text: string;
}

export default function Chat() {
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const scrollAnchor = useRef<HTMLDivElement>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    const que = inputRef.current?.value.trim() || "";
    if (!que) return;

    setMessages((prev) => [...prev, { sender: "user", text: que }]);
    inputRef.current!.value = "";
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: que }),
      });

      const data = await response.json();
      const answer = data?.answer || "No response";

      setMessages((prev) => [...prev, { sender: "bot", text: answer }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Error reaching backend." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  useEffect(() => {
    scrollAnchor.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className={styles.chatApp}>
      <header className={styles.appHeader}>
        <h1>Covid Smoking Agent<span>Bot</span></h1>
      </header>

      <div className={styles.chatContainer}>
        <div className={styles.messagesContainer}>
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`${styles.messageWrapper} ${
                msg.sender === "user" ? styles.userWrapper : ""
              }`}
            >
              <div className={styles.messageContainer}>
                <div
                  className={`${styles.message} ${
                    msg.sender === "user"
                      ? styles.userMessage
                      : styles.botMessage
                  }`}
                >
                  <div className={styles.messageContent}>{msg.text}</div>
                </div>
              </div>
            </div>
          ))}

          {loading && (
            <div className={`${styles.messageWrapper}`}>
              <div className={styles.messageContainer}>
                <div className={`${styles.message} ${styles.botMessage}`}>
                  <div className={`${styles.messageContent}`}>
                    <div className={styles.typingIndicator}>
                      <div className={styles.dot}></div>
                      <div className={styles.dot}></div>
                      <div className={styles.dot}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={scrollAnchor} className={styles.scrollAnchor}></div>
        </div>

        <form
          className={styles.inputContainer}
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage();
          }}
        >
          <div className={styles.inputForm}>
            <textarea
              ref={inputRef}
              className={styles.inputField}
              placeholder="Type your message..."
              onKeyDown={handleKeyDown}
              rows={1}
            />
            <button type="submit" className={styles.sendButton}>
              <PaperPlaneIcon width={20} height={20} />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
