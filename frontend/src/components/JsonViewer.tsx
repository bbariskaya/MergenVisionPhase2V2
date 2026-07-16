function JsonViewer({ data }: { data: unknown }) {
  return (
    <pre className="max-h-[28rem] overflow-auto rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] p-4 text-xs leading-relaxed text-[var(--color-foreground)]">
      <code>{JSON.stringify(data, null, 2)}</code>
    </pre>
  )
}

export default JsonViewer
