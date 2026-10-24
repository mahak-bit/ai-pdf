"use client";

import { useEffect, useRef, useState } from "react";
import { SendIcon } from "./icons";

const MAX_CHARS = 1000;

type Props = {
  disabled: boolean;
  busy: boolean;
  placeholder: string;
  onSubmit: (question: string) => void;
};

export function Composer({ disabled, busy, placeholder, onSubmit }: Props) {
  const [value, setValue] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);
  const trimmed = value.trim();
  const canSend = !disabled && !busy && trimmed.length > 0 && value.length <= MAX_CHARS;

  useEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [value]);

  const submit = () => {
    if (!canSend) return;
    onSubmit(trimmed);
    setValue("");
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      className="rounded-2xl border border-line-strong bg-paper p-2 transition-colors focus-within:border-blue"
    >
      <label htmlFor="question" className="sr-only">
        Ask a question about the document
      </label>
      <div className="flex items-end gap-2">
        <textarea
          id="question"
          ref={area}
          rows={1}
          value={value}
          disabled={disabled}
          placeholder={placeholder}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          className="max-h-[180px] min-h-10 flex-1 resize-none bg-transparent px-2.5 py-2 outline-none placeholder:text-mute focus-visible:outline-none disabled:cursor-not-allowed"
        />
        <button
          type="submit"
          disabled={!canSend}
          aria-label="Ask"
          className="grid size-10 shrink-0 place-items-center rounded-xl bg-blue text-white transition-colors hover:bg-blue-deep disabled:bg-line disabled:text-mute"
        >
          <SendIcon />
        </button>
      </div>
      {value.length > MAX_CHARS * 0.8 && (
        <p className={`px-2.5 pt-1 text-xs ${value.length > MAX_CHARS ? "text-danger" : "text-mute"}`}>
          {value.length}/{MAX_CHARS} characters
        </p>
      )}
    </form>
  );
}
