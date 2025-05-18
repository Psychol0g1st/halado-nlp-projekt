"use client";

import React, { useEffect } from "react";
import { Button } from "../ui/button";
import { AnimatePresence } from "framer-motion";
import { ChatProps } from "@/lib/types";
import useChatStore from "@/hooks/useChatStore";
import {SendHorizonal } from "lucide-react";
import useSpeechToText from "@/hooks/useSpeechRecognition";
import { ChatInput } from "../ui/chat/chat-input";
import { useState } from "react";
interface MergedProps extends ChatProps {
  files: File[] | undefined;
  onAsk: (question : string) => void;
  onResponse: (responseText: string) => void;
  onLoading: (loading: boolean) => void;
}

export default function ChatBottombar({
  stop,
  files,
  onAsk,
  onResponse,
  onLoading
}: MergedProps) {
  const input = useChatStore((state) => state.input);
  const handleInputChange = useChatStore((state) => state.handleInputChange);
  const inputRef = React.useRef<HTMLTextAreaElement>(null);

  const isLoading = useChatStore((state) => state.isLoading);
  const setInput = useChatStore((state) => state.setInput);
  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      if (isLoading) return;

      e.preventDefault();
      handleSubmit();
    }
  };

  const { isListening, transcript, startListening, stopListening } =
    useSpeechToText({ continuous: true });

  const listen = () => {
    isListening ? stopVoiceInput() : startListening();
  };

  const stopVoiceInput = () => {
    setInput(transcript.length ? transcript : "");
    stopListening();
  };

   const handleSubmit = async() => {
    try {
      onAsk(input)
      onLoading(true);
      setInput("");
      const que = inputRef.current?.value || "";
      const response = await fetch('http://localhost:8000/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: que }),
          });


        if (!response.ok) {
            return "Error: " + response.statusText;
        }

        const data = await response.json();
        //console.log(data.answer);
        const answer = data.answer;
        onResponse(answer);
        onLoading(false);

      } catch (error) {
        console.error('Error:', error);
      }
  }


  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  useEffect(() => {
    if (isLoading) {
      stopVoiceInput();
    }
  }, [isLoading]);

  return (
    <div className="px-4 pb-7 flex justify-between w-full items-center relative ">
          <ChatInput
            autoComplete="off"
            value={isListening ? (transcript.length ? transcript : "") : input}
            ref={inputRef}
            onKeyDown={handleKeyPress}
            onChange={handleInputChange}
            name="message"
            placeholder={!isListening ? "Enter your prompt here" : "Listening"}
            className="max-h-40 px-6 pt-6 border-0 shadow-none bg-accent rounded-lg text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-0 disabled:cursor-not-allowed dark:bg-card"
          />

          <Button
            className="shrink-0 rounded-full absolute right-[20px]"
            variant="ghost"
            onClick={handleSubmit}
            >

            <SendHorizonal className="w-5 h-5" />
          </Button>
    </div>
  );
}
