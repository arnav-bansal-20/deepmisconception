import React, { useRef } from 'react';
import Editor, { OnMount, OnChange } from '@monaco-editor/react';
import { ProcessEventItem } from '../types';

interface MonacoEditorProps {
  value: string;
  onChange: (value: string) => void;
  onTelemetryEvent?: (event: ProcessEventItem) => void;
  readOnly?: boolean;
}

export const MonacoEditor: React.FC<MonacoEditorProps> = ({
  value,
  onChange,
  onTelemetryEvent,
  readOnly = false,
}) => {
  const lastLengthRef = useRef<number>(value.length);
  const editorRef = useRef<any>(null);

  const handleEditorDidMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;

    // Listen to paste events in Monaco
    editor.onDidPaste((pasteEvent) => {
      const startLineNumber = pasteEvent.range.startLineNumber;
      const endLineNumber = pasteEvent.range.endLineNumber;
      const model = editor.getModel();
      const pastedText = model ? model.getValueInRange(pasteEvent.range) : '';

      if (onTelemetryEvent) {
        onTelemetryEvent({
          event_type: 'paste',
          cursor_line: startLineNumber,
          cursor_column: pasteEvent.range.startColumn,
          delta_chars: pastedText.length,
          paste_length: pastedText.length,
          timestamp: new Date().toISOString(),
          event_metadata: {
            lineSpan: endLineNumber - startLineNumber,
          },
        });
      }
    });

    // Listen to cursor position changes
    editor.onDidChangeCursorPosition((e) => {
      // Optional fine-grained telemetry
    });
  };

  const handleEditorChange: OnChange = (newVal = '') => {
    onChange(newVal);
    const delta = newVal.length - lastLengthRef.current;
    lastLengthRef.current = newVal.length;

    if (onTelemetryEvent && Math.abs(delta) > 0) {
      const position = editorRef.current?.getPosition() || { lineNumber: 1, column: 1 };
      onTelemetryEvent({
        event_type: 'edit',
        cursor_line: position.lineNumber,
        cursor_column: position.column,
        delta_chars: delta,
        paste_length: 0,
        timestamp: new Date().toISOString(),
      });
    }
  };

  return (
    <div className="w-full h-full rounded-xl overflow-hidden border border-slate-300 shadow-sm bg-white">
      <Editor
        height="100%"
        defaultLanguage="python"
        theme="vs"
        value={value}
        onChange={handleEditorChange}
        onMount={handleEditorDidMount}
        options={{
          readOnly,
          minimap: { enabled: false },
          fontSize: 14,
          lineNumbers: 'on',
          scrollBeyondLastLine: false,
          automaticLayout: true,
          tabSize: 4,
          insertSpaces: true,
          fontFamily: "'Fira Code', 'JetBrains Mono', Consolas, monospace",
          renderWhitespace: 'selection',
          padding: { top: 12, bottom: 12 },
        }}
      />
    </div>
  );
};
