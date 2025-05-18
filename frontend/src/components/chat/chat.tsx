"use client";

import React from "react";
import ChatList from "./chat-list";
import ChatBottombar from "./chat-bottombar";
import { useChat } from "@/hooks/useChat";
import { MessageWithFiles } from "@/lib/types";
import Image from "next/image";
import AttachedFiles from "../attached-files";
import { useState } from "react";
import { string } from "zod";
interface ChatProps {
  id: string;
  initialMessages: MessageWithFiles[];
}

export default function Chat({ initialMessages, id }: ChatProps) {
  const {
    loadingSubmit,
    handleSubmit,
    stop,
    regenerate,
    files,
    setFiles,
    fileText,
    setFileText,
    open,
    setOpen,
  } = useChat({ id, initialMessages });
 const [messages, setMessages] = useState<string[]>([]);
 const [loading, setLoading] = useState<boolean>(false);


  const handleNewAsk = (question:string) => {
    setMessages((prev) => [...prev, question]);
  };

  const handleNewMessage = (responseText: string) => {
    setMessages((prev) => [...prev,responseText]);
  };
  
  const handleLoading = (loading: boolean) => {
    setLoading(loading);
  };
  
  
  

  return (
    <div className="flex flex-col justify-items-end w-full max-w-3xl h-full">
      {messages.length === 0 ? (
        <div className="flex flex-col h-full w-full items-center gap-4 justify-end pb-[50px]">
          <div className="flex flex-col gap-1 items-center">
            <Image
              src="/logo.svg"
              alt="AI"
              width={70}
              height={70}
              className="dark:invert"
            />
            <p className="text-center text-2xl md:text-3xl font-semibold text-muted-foreground/75">
              How can I help you today?
            </p>
            <p className="text-center text-sm text-muted-foreground/75 max-w-lg">
              Models with <strong>(1k)</strong> suffix lowers VRAM requirements
              by ~2-3GB.
            </p>
          </div>

          <div className="flex flex-col w-full ">
            <ChatBottombar
              files={files}
              stop={stop}
              messages={messages}
              onAsk={handleNewAsk}
              onResponse={handleNewMessage}
              onLoading={handleLoading}
            />
          </div>
        </div>
      ) : (
        <>
          <ChatList
            messages={messages}
            stop={stop}
            chatId={id}
            loadingSubmit={loading}
            onRegenerate={regenerate}
          />
          <ChatBottombar
            files={files}
            stop={stop}
            messages={messages}
            onAsk={handleNewAsk}
            onResponse={handleNewMessage}
            onLoading={handleLoading}
          />
        </>
      )}
    </div>
  );
}
