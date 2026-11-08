interface UploadResult {
  processed?: unknown[];
  errors?: { filename: string; error: string }[];
}

export function uploadFeedback(result: UploadResult): {
  type: "success" | "error";
  text: string;
} {
  const count = result.processed?.length ?? 0;
  const failures = result.errors ?? [];
  if (failures.length > 0) {
    return {
      type: "error",
      text: `Uploaded ${count} file(s). ${failures.length} file(s) could not be processed: ${failures.map((item) => `${item.filename}: ${item.error}`).join("; ")}. Retry only the failed files.`,
    };
  }
  return count > 0
    ? { type: "success", text: `Successfully uploaded ${count} file(s)!` }
    : {
        type: "error",
        text: "No files were processed. Choose your files again to retry.",
      };
}
