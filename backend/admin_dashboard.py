from datetime import datetime


def render_admin_dashboard(
    interactions_list: list,
    faces_list: list,
    sessions_list: list,
    profile_list: list,
) -> str:
    """Render the admin dashboard HTML for the logs-dashboard admin route."""

    interaction_rows = ""
    for idx, item in enumerate(interactions_list):
        ts = item.get("timestamp")
        time_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts or "N/A")
        sess_id = item.get("session_id", "N/A")
        interaction_rows += f"""
        <tr id=\"interaction-{sess_id}\">
            <td>{idx + 1}</td>
            <td><code>{sess_id}</code></td>
            <td><strong>{item.get('input_text', 'N/A')}</strong></td>
            <td>{item.get('response_text', 'N/A')}</td>
            <td><span class=\"badge\">{time_str}</span></td>
            <td>
                <button class=\"btn btn-danger\" onclick=\"deleteInteraction('{sess_id}')\">Delete Log</button>
            </td>
        </tr>
        """

    face_rows = ""
    for idx, item in enumerate(faces_list):
        ts = item.get("detected_at") or item.get("last_seen")
        time_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts or "N/A")
        face_id = item.get("face_id", "N/A")
        current_name = item.get("name", "Unknown Visitor")
        face_rows += f"""
        <tr id=\"face-{face_id}\">
            <td>{idx + 1}</td>
            <td><code>{face_id}</code></td>
            <td><strong id=\"face-name-text-{face_id}\">{current_name}</strong></td>
            <td>{item.get('visit_count', 1)}</td>
            <td><span class=\"badge\">{time_str}</span></td>
            <td>
                <button class=\"btn btn-edit\" onclick=\"editFaceName('{face_id}', '{current_name}')\">Rename</button>
                <button class=\"btn btn-danger\" onclick=\"deleteFace('{face_id}')\">Delete</button>
            </td>
        </tr>
        """

    session_rows = ""
    for idx, item in enumerate(sessions_list):
        ts = item.get("start_time")
        time_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts or "N/A")
        sess_id = item.get("session_id", "N/A")
        session_rows += f"""
        <tr id=\"session-{sess_id}\">
            <td><code>{sess_id}</code></td>
            <td>{item.get('user_name', 'Guest')}</td>
            <td>{item.get('visit_count', 1)}</td>
            <td><span class=\"badge\">{time_str}</span></td>
            <td>
                <button class=\"btn btn-danger\" onclick=\"deleteSession('{sess_id}')\">End & Delete</button>
            </td>
        </tr>
        """

    profile_rows = ""
    for idx, item in enumerate(profile_list):
        profile_rows += f"""
        <tr>
            <td>{idx + 1}</td>
            <td><span class=\"badge\" style=\"background:#0066cc; color:white;\">{item.get('category', 'General')}</span></td>
            <td><strong>{item.get('question_or_key', item.get('question', 'N/A'))}</strong></td>
            <td>{item.get('fact_details', item.get('answer', 'N/A'))}</td>
            <td><span class=\"badge\" style=\"background:#e2e8f0; color:#475569;\">Static</span></td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <title>RNSIT Kiosk - Admin Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 30px; background-color: #f4f6f9; color: #333; }}
            .container {{ max-width: 1300px; margin: 0 auto; }}
            .header-flex {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
            h1 {{ color: #0066cc; margin: 0; font-size: 28px; }}
            .subtitle {{ color: #666; margin-top: 5px; margin-bottom: 30px; font-size: 15px; }}
            .tabs {{ display: flex; flex-wrap: wrap; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }}
            .tabs label {{ padding: 15px 25px; cursor: pointer; background: #f8fafc; font-weight: 600; border-bottom: 3px solid transparent; transition: ease 0.2s; order: 1; }}
            .tabs input[type="radio"] {{ display: none; }}
            .tab-content {{ width: 100%; padding: 25px; background: #fff; border-top: 1px solid #e2e8f0; display: none; order: 99; overflow-x: auto; }}
            .tabs input[type="radio"]:checked + label {{ border-bottom: 3px solid #0066cc; background: #fff; color: #0066cc; }}
            .tabs input[type="radio"]:checked + label + .tab-content {{ display: block; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; min-width: 800px; }}
            th, td {{ padding: 14px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
            th {{ background-color: #f8fafc; color: #475569; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; font-weight: 700; }}
            tr:hover {{ background-color: #f8fafc; }}
            code {{ background: #f1f5f9; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-family: Courier, monospace; color: #0066cc; }}
            .badge {{ background: #e2e8f0; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
            .btn {{ padding: 6px 12px; border: none; border-radius: 4px; font-size: 12px; font-weight: 600; cursor: pointer; transition: 0.2s ease; margin-right: 5px; }}
            .btn-danger {{ background-color: #fee2e2; color: #dc2626; }}
            .btn-danger:hover {{ background-color: #fca5a5; }}
            .btn-edit {{ background-color: #e0f2fe; color: #0284c7; }}
            .btn-edit:hover {{ background-color: #bae6fd; }}
            .btn-reset {{ background-color: #dc2626; color: white; padding: 10px 20px; font-size: 14px; border-radius: 6px; box-shadow: 0 4px 6px rgba(220, 38, 38, 0.2); }}
            .btn-reset:hover {{ background-color: #b91c1c; }}
        </style>
    </head>
    <body>
        <div class=\"container\">
            <div class=\"header-flex\">
                <div>
                    <h1>RNSIT Kiosk - Admin Dashboard</h1>
                    <p class=\"subtitle\">Secure Read/Write control plane managing collections, tracked users, and kiosk interactions.</p>
                </div>
                <button class=\"btn btn-reset\" onclick=\"clearAllTestData()\">🔄 Clear All Sessions & Interactions</button>
            </div>
            <div class=\"tabs\">
                <input type=\"radio\" name=\"admin_tabs\" id=\"tab_interactions\" checked>
                <label for=\"tab_interactions\"> Interactions ({len(interactions_list)})</label>
                <div class=\"tab-content\">
                    <h3> Live Interactions Log (`interactions` collection)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th style=\"width: 5%\">#</th>
                                <th style=\"width: 15%\">Session ID</th>
                                <th style=\"width: 25%\">User Query</th>
                                <th style=\"width: 35%\">Kiosk Response</th>
                                <th style=\"width: 12%\">Timestamp</th>
                                <th style=\"width: 8%\">Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {interaction_rows if interaction_rows else "<tr><td colspan='6' style='text-align:center;'>No interactions recorded yet.</td></tr>"}
                        </tbody>
                    </table>
                </div>
                <input type=\"radio\" name=\"admin_tabs\" id=\"tab_faces\">
                <label for=\"tab_faces\"> Face Tracks ({len(faces_list)})</label>
                <div class=\"tab-content\">
                    <h3> Registered Facial Profiles (`faces` collection)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th style=\"width: 5%\">#</th>
                                <th style=\"width: 25%\">Face ID Token</th>
                                <th style=\"width: 25%\">Identified Name</th>
                                <th style=\"width: 15%\">Visit Count</th>
                                <th style=\"width: 18%\">Last Spotted</th>
                                <th style=\"width: 12%\">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {face_rows if face_rows else "<tr><td colspan='6' style='text-align:center;'>No facial profiles tracked yet.</td></tr>"}
                        </tbody>
                    </table>
                </div>
                <input type=\"radio\" name=\"admin_tabs\" id=\"tab_sessions\">
                <label for=\"tab_sessions\"> Active Sessions ({len(sessions_list)})</label>
                <div class=\"tab-content\">
                    <h3>Session Registries (`sessions` collection)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th style=\"width: 25%\">Session ID</th>
                                <th style=\"width: 25%\">Visitor Name</th>
                                <th style=\"width: 20%\">Session Visit Count</th>
                                <th style=\"width: 18%\">Started At</th>
                                <th style=\"width: 12%\">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {session_rows if session_rows else "<tr><td colspan='5' style='text-align:center;'>No active sessions.</td></tr>"}
                        </tbody>
                    </table>
                </div>
                <input type=\"radio\" name=\"admin_tabs\" id=\"tab_profile\">
                <label for=\"tab_profile\"> Knowledge Base ({len(profile_list)})</label>
                <div class=\"tab-content\">
                    <h3>College Profile Documents (`college_profile` collection)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th style=\"width: 5%\">#</th>
                                <th style=\"width: 15%\">Category</th>
                                <th style=\"width: 25%\">Topic Key / FAQ Question</th>
                                <th style=\"width: 45%\">Stored Fact Details / FAQ Answer</th>
                                <th style=\"width: 10%\">Type</th>
                            </tr>
                        </thead>
                        <tbody>
                            {profile_rows if profile_rows else "<tr><td colspan='5' style='text-align:center;'>Knowledge base is empty.</td></tr>"}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <script>
            async function makeRequest(url, method, body = null) {{
                const headers = {{
                    "Content-Type": "application/json",
                    "Authorization": "Basic " + btoa("admin:111111")
                }};
                const config = {{ method, headers }};
                if (body) {{ config.body = JSON.stringify(body); }}
                try {{
                    const response = await fetch(url, config);
                    if (!response.ok) {{
                        const errorData = await response.json();
                        throw new Error(errorData.detail || "Server error occurred");
                    }}
                    return await response.json();
                }} catch (err) {{
                    alert("Operation failed: " + err.message);
                    return null;
                }}
            }}
            async function deleteInteraction(sessionId) {{
                if (confirm(`Do you want to purge interaction logs for session: ${{sessionId}}?`)) {{
                    const result = await makeRequest(`/api/admin/interactions/${{sessionId}}`, "DELETE");
                    if (result) {{
                        alert(result.message);
                        location.reload();
                    }}
                }}
            }}
            async function deleteFace(faceId) {{
                if (confirm(`Are you sure you want to delete profile ${{faceId}}? This action resets their history.`)) {{
                    const result = await makeRequest(`/api/admin/faces/${{faceId}}`, "DELETE");
                    if (result) {{
                        alert(result.message);
                        document.getElementById(`face-${{faceId}}`)?.remove();
                    }}
                }}
            }}
            async function editFaceName(faceId, currentName) {{
                const newName = prompt(`Enter a new display name for ${{currentName}}:`, currentName);
                if (newName && newName.trim() !== "" && newName !== currentName) {{
                    const result = await makeRequest(`/api/admin/faces/${{faceId}}`, "PUT", {{ name: newName.trim() }});
                    if (result) {{
                        document.getElementById(`face-name-text-${{faceId}}`).textContent = newName.trim();
                    }}
                }}
            }}
            async function deleteSession(sessionId) {{
                if (confirm(`Terminate registry profile for session ID ${{sessionId}}?`)) {{
                    const result = await makeRequest(`/api/admin/sessions/${{sessionId}}`, "DELETE");
                    if (result) {{
                        alert("Session successfully dropped.");
                        document.getElementById(`session-${{sessionId}}`)?.remove();
                    }}
                }}
            }}
            async function clearAllTestData() {{
                if (confirm("MASTER DESTRUCTION WARNING: This clears all user sessions and live interaction records. Are you sure you want to clean up for a new presentation run?")) {{
                    const result = await makeRequest("/api/admin/clear-all", "DELETE");
                    if (result) {{
                        alert(result.message);
                        location.reload();
                    }}
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html_content
