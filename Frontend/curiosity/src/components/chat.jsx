"use client";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import io from "socket.io-client";
import { v4 as uuidv4 } from "uuid";

const socket = io("http://localhost:4000");

const Chat = () => {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);

  // Listen for "sources" events
  useEffect(() => {
    socket.on("sources", (data) => {
      // data: { id, sources }
      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.id === data.id && msg.type === "received") {
            return { ...msg, sources: data.sources };
          }
          return msg;
        })
      );
    });

    // Listen for "message" events
    socket.on("message", (data) => {
      // data: { id, text }
      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.id === data.id && msg.type === "received") {
            return { ...msg, text: data.text };
          }
          return msg;
        })
      );
    });

    return () => {
      socket.off("sources");
      socket.off("message");
    };
  }, []);

  const sendMessage = () => {
    if (message.trim() === "") return;
    const id = uuidv4();

    // Add both the sent message and a placeholder received message
    setMessages((prev) => [
      ...prev,
      { id, text: message, type: "sent" },
      { id, text: "Waiting for response...", type: "received", sources: [] },
    ]);

    // Emit the message with its unique ID to the backend
    socket.emit("message", { id, text: message });
    setMessage("");
  };

  return (
    <>
      <div className="w-full flex flex-col gap-5 mb-24">
        {messages.map((msg) =>
          msg.type === "sent" ? (
            <SentMessage key={`${msg.id}-sent`} message={msg.text} />
          ) : (
            <ReceivedMessage
              key={`${msg.id}-received`}
              message={msg.text}
              sources={msg.sources}
            />
          )
        )}
      </div>
      <div className="fixed bottom-0 w-full text-white p-4 flex space-x-3 border-t border-input bg-background">
        <Input
          type="text"
          placeholder="How can Curiosity help you today?"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <Button type="submit" onClick={sendMessage}>
          Send
        </Button>
      </div>
    </>
  );
};

const SentMessage = ({ message }) => {
  return (
    <div className="self-end p-2 m-2 border border-input max-w-sm break-words">
      {message}
    </div>
  );
};

const ReceivedMessage = ({ message, sources }) => {
  return (
    <div className="flex flex-col gap-2 self-start p-2 m-2 border-t border-input min-w-[90%] break-words">
      <div>
        {sources && sources.length > 0 ? (
          <h1 className="m-1 font-bold">Sources</h1>
        ) : null}
        <div className="flex flex-wrap gap-2">
          {sources && sources.length > 0
            ? sources.map((url, index) => <Sources key={index} url={url} />)
            : null}
        </div>
        <div className="break-words p-2 font-medium">{message}</div>
      </div>
    </div>
  );
};

const Sources = ({ url }) => {
  const parsedUrl = url.replace(/^https?:\/\/www\./, "").split(".")[0];
  const parsedUrlForFavicon = url
    .replace(/^https?:\/\/(www\.)?/, "")
    .split("/")[0];
  return (
    <div className="flex items-center gap-2 border border-input p-1">
      <img
        src={`https://www.google.com/s2/favicons?domain=${parsedUrlForFavicon}`}
        alt="favicon"
        className="w-[20px] h-[20px] object-cover"
      />
      {parsedUrl}
    </div>
  );
};

export default Chat;
