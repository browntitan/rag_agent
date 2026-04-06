import { useState, useCallback } from 'react'
import { getErrorMessage, isBackendUnavailableError, uploadFiles } from '../api/client'
import type { UploadResult } from '../types'

export function useFileUpload(
  conversationId: string,
  onBackendUnavailable?: (message: string) => void,
) {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([])
  const [lastError, setLastError] = useState<string | null>(null)

  const upload = useCallback(async (files: File[]): Promise<UploadResult | null> => {
    if (files.length === 0) return null
    setIsUploading(true)
    setLastError(null)

    try {
      const result = await uploadFiles(files, conversationId)
      setUploadedFiles(prev => [...prev, ...result.filenames])
      if (result.errors.length > 0) {
        setLastError(result.errors.join(', '))
      }
      return result
    } catch (error) {
      const msg = getErrorMessage(error)
      if (isBackendUnavailableError(error)) {
        onBackendUnavailable?.(msg)
      }
      setLastError(msg)
      return null
    } finally {
      setIsUploading(false)
    }
  }, [conversationId, onBackendUnavailable])

  return { isUploading, uploadedFiles, lastError, upload }
}
