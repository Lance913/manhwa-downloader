' Double-click this file to open the Manhwa Panel Downloader - no console
' window, no typing. Finds its own folder, so it keeps working after
' being moved as long as it stays next to gui.py and .venv.

Set fso = CreateObject("Scripting.FileSystemObject")
projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = projectDir & "\.venv\Scripts\pythonw.exe"

If Not fso.FileExists(pythonw) Then
    MsgBox "Setup hasn't been run yet." & vbCrLf & vbCrLf & _
           "Double-click setup.bat first, then try this again.", _
           vbExclamation, "Manhwa Panel Downloader"
    WScript.Quit
End If

Q = Chr(34)
guiScript = projectDir & "\gui.py"
cmd = Q & pythonw & Q & " " & Q & guiScript & Q

Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = projectDir
shell.Run cmd, 0, False
