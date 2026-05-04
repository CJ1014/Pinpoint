[Setup]
AppName=PinPoint
AppVersion={#AppVersion}
AppPublisher=CJ
AppPublisherURL=https://github.com/cj1014/Pinpoint
AppSupportURL=https://github.com/cj1014/Pinpoint
DefaultDirName={autopf}\PinPoint
DefaultGroupName=PinPoint
OutputBaseFilename=PinPoint-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "dist\PinPoint.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\PinPoint"; Filename: "{app}\PinPoint.exe"
Name: "{group}\Uninstall PinPoint"; Filename: "{uninstallexe}"
Name: "{commondesktop}\PinPoint"; Filename: "{app}\PinPoint.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\PinPoint.exe"; Description: "Launch PinPoint now"; Flags: nowait postinstall skipifsilent

[Messages]
WelcomeLabel2=This will install PinPoint on your computer.%n%nPinPoint requires Ollama to be installed and running. Get it free at https://ollama.com%n%nRecommended model: ollama pull llama3.3
