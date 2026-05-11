# GitHub Actions Self-Hosted Runner - Configuração

## 📋 Pré-requisitos

- [x] Git instalado
- [x] Python 3.12+ instalado
- [x] Conta GitHub com acesso ao repositório
- [x] Windows PowerShell ou Command Prompt

## 🚀 Passo 1: Gerar Token de Acesso Pessoal

1. Vá para **GitHub.com** > Settings > Developer settings > Personal access tokens > Tokens (classic)
2. Clique em "Generate new token (classic)"
3. Marque as permissões:
   - `repo` (acesso completo ao repositório)
   - `workflow` (permissão para workflows)
   - `admin:repo_hook` (webhooks)
4. Copie o token (você usará isso em breve)

## 🔧 Passo 2: Registrar Self-Hosted Runner

### Via PowerShell (Windows):

```powershell
# 1. Crie uma pasta para o runner
mkdir "C:\GitHub\Actions\runner"
cd "C:\GitHub\Actions\runner"

# 2. Download o runner (veja a versão mais recente em github.com/actions/runner/releases)
$latestRelease = (Invoke-WebRequest -Uri "https://api.github.com/repos/actions/runner/releases/latest" | ConvertFrom-Json).tag_name
$version = $latestRelease -replace "v", ""
Invoke-WebRequest -Uri "https://github.com/actions/runner/releases/download/$latestRelease/actions-runner-win-x64-$version.zip" -OutFile "runner.zip"

# 3. Descompacte
Expand-Archive -Path "runner.zip" -DestinationPath "."
Remove-Item "runner.zip"

# 4. Configure o runner (você precisará do seu TOKEN)
.\config.cmd --url https://github.com/henriquefmoura/machinelearning --token $token
```

**Quando solicitado:**
- Nome do runner: `win-runner-001`
- Grupo de runner: deixe em branco (padrão)
- Diretório de trabalho: deixe em branco (padrão)

## 🔄 Passo 3: Executar o Runner como Serviço (Recomendado)

Para que o runner execute **mesmo quando você não está logado**:

```powershell
# Com privilégios de administrador:
.\svc.cmd install "NT AUTHORITY\SYSTEM"
.\svc.cmd start
```

Verifique o status:
```powershell
.\svc.cmd status
```

## ✅ Passo 4: Validar Configuração

1. Vá para **GitHub.com** > seu repositório > Settings > Actions > Runners
2. Você deve ver `win-runner-001` com status **Idle** (pronto)

## 🎯 Testar o Workflow

Execute manualmente:

```powershell
# No GitHub:
# Vá para Actions > "RPA - Busca Artigos a Cada Hora" > "Run workflow" > Run workflow
```

Ou aguarde a próxima hora (o workflow roda automaticamente no horário 0 de cada hora UTC).

## 📍 Localização dos Arquivos

O runner salvará tudo localmente:
- **PDFs**: `c:\Users\Henrique F. Moura\OneDrive\Documentos\doutorado\machinelearning\artigos_pdf_12_meses\pdfs\`
- **CSVs**: `c:\Users\Henrique F. Moura\OneDrive\Documentos\doutorado\machinelearning\artigos_pdf_12_meses\`
- **Logs**: `C:\GitHub\Actions\runner\_diag\`

## 🔐 Segurança

⚠️ **IMPORTANTE**: Guarde seu token de acesso com segurança. Não compartilhe com ninguém!

Se o token for comprometido:
1. Acesse GitHub Settings > Developer settings > Personal access tokens
2. Clique em "Delete"
3. Gere um novo token
4. Reconfigure o runner: `.\config.cmd --remove`

## ❌ Parar o Runner

```powershell
# Se instalado como serviço:
.\svc.cmd stop
.\svc.cmd uninstall

# Ou simplesmente execute:
.\run.cmd  # Isso abre o runner em uma janela que você pode fechar
```

## 📊 Monitorar Execuções

No GitHub:
- **Actions** > **RPA - Busca Artigos a Cada Hora**
- Veja logs, duração, e status de cada execução

## 🆘 Troubleshooting

### Runner aparece "Offline"
```powershell
# Verifique se o serviço está rodando
Get-Service -Name "GitHub.Runner*" | Select Status

# Reinicie o serviço
Restart-Service -Name "GitHub.Runner*"
```

### Erros de Permissão
- Certifique-se que a conta que roda o serviço tem acesso aos diretórios
- Execute PowerShell como Administrador para instalar/configurar

### Python/Dependências não encontradas
```powershell
# Adicione ao seu PATH do Windows:
# C:\Users\Henrique F. Moura\OneDrive\Documentos\doutorado\machinelearning\.venv\Scripts

# Ou execute manualmente no runner:
python -m pip install -r requirements.txt
```

---

**Pronto!** Seu RPA agora roda automaticamente na nuvem GitHub, mas salva tudo localmente no seu OneDrive! 🎉
