import { useRef, type ChangeEvent } from 'react'
import './FileUpload.css'

interface Props {
  isUploading: boolean
  uploadedFiles: string[]
  lastError: string | null
  disabled?: boolean
  onUpload: (files: File[]) => void
}

const ACCEPT = '.pdf,.docx,.txt,.csv,.xlsx,.md'

export function FileUpload({ isUploading, uploadedFiles, lastError, disabled, onUpload }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      onUpload(Array.from(files))
      e.target.value = ''  // allow re-uploading same file
    }
  }

  return (
    <div className="file-upload">
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPT}
        className="file-upload__input"
        disabled={disabled || isUploading}
        onChange={handleChange}
      />
      <button
        className="file-upload__btn"
        onClick={() => inputRef.current?.click()}
        disabled={disabled || isUploading}
      >
        {isUploading ? 'Uploading...' : 'Upload Files'}
      </button>
      {uploadedFiles.length > 0 && (
        <span className="file-upload__badge">{uploadedFiles.length}</span>
      )}
      {lastError && <span className="file-upload__error">{lastError}</span>}
    </div>
  )
}
