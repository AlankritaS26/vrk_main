# RNSIT Staff Mobile App

This native Expo Go app is the staff companion for the RNSIT digital receptionist kiosk. It uses the phone's native microphone/audio permissions instead of browser microphone permissions. It uses `expo-audio`, not the older `expo-av` module.

## What it does

- Shows active human-escalation requests and recent history.
- Accepts and resolves visitor escalations.
- Records five-second staff voice clips.
- Sends staff audio for transcription and relays the original staff voice to the kiosk.
- Receives and plays the visitor's original voice on the staff phone.
- Receives live transcript/audio updates over WebSocket.
- Automatically scrolls active conversations to the newest message.

## Requirements

- Expo Go on the staff phone.
- Backend running on port `8001`.
- Phone and backend computer on the same Wi-Fi.
- Backend LAN address, such as `192.168.0.117`.

## Run it

From the repository root:

```powershell
Set-Location "C:\Users\Akshatha A\vrk_main\staff-mobile"
npm install
npx expo start --lan --port 8086
```

If the port is busy, use another free port. To clear stale Metro/Expo Go cache:

```powershell
npx expo start --lan --port 8086 --clear
```

Then:

1. Open Expo Go and scan the QR code.
2. Enter `http://192.168.0.117:8001`.
3. Log in with username `staff` and password `rnsit2024`.
4. Tap **Connect**.
5. Accept an active escalation.
6. Tap **Turn microphone on**.
7. Tap **Turn microphone off** when finished.
8. Tap **Resolve** to close the escalation.

The phone and backend PC must be on the same Wi-Fi.

## Audio behavior

- Staff phone voice plays on the kiosk as original recorded audio.
- Visitor kiosk voice plays on the connected staff phone.
- Visitor voice is not played back on the kiosk, preventing echo.
- Text-only staff replies use the kiosk's Kokoro TTS endpoint.
- Existing kiosk STT remains primary: faster-whisper through `/stt/pcm`.

## Development notes

- Do not run `cd staff-mobile` when already inside `staff-mobile`.
- Do not use `CI=1` during normal development.
- The Connect screen loads the active queue first and history in the background.
- The app connects to the backend WebSocket after login for live audio and messages.

If Expo Go still shows the old `unsupported FormData implementation` message, stop the Expo server, then restart it from this folder with:

```powershell
npx expo start --lan --port 8086 --clear
```

Reload the project in Expo Go after the QR code appears. The current app uses Expo's native multipart file uploader rather than JavaScript `FormData`.

### Duplicate `setAudioModeAsync` import

Expo Go is using an old cached bundle. Stop Metro, restart with `--clear`, fully close Expo Go, and scan the new QR code.

### Cannot connect

Check that the backend is running on port `8001`, replace `127.0.0.1` with the backend computer's LAN IP, keep both devices on the same Wi-Fi, and check Windows Firewall.

### Visitor audio is silent

Confirm the escalation is accepted and connected, raise the phone volume, and ensure the phone is not in silent mode.
