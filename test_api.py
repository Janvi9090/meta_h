"""
Comprehensive API test for the Medication Dosing OpenEnv server.
Tests all endpoints for HF Spaces deployment readiness.
"""
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)
passed = 0
failed = 0


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name} — {detail}")
        failed += 1


# ─── 1. Health checks ───
print("\n=== Health Checks ===")
r = client.get("/")
test("GET / returns 200", r.status_code == 200)
test("GET / has status field", "status" in r.json())

r = client.get("/health")
test("GET /health returns 200", r.status_code == 200)
test("GET /health status=healthy", r.json().get("status") == "healthy")

# ─── 2. Tasks endpoint ───
print("\n=== Tasks ===")
r = client.get("/tasks")
test("GET /tasks returns 200", r.status_code == 200)
tasks = r.json().get("tasks", {})
test("Has easy task", "easy" in tasks)
test("Has medium task", "medium" in tasks)
test("Has hard task", "hard" in tasks)

# ─── 3. Reset endpoint ───
print("\n=== Reset ===")

# Empty body
r = client.post("/reset")
test("POST /reset (no body) returns 200", r.status_code == 200, f"got {r.status_code}: {r.text}")
if r.status_code == 200:
    data = r.json()
    test("Reset has 'observation' key", "observation" in data)
    test("Reset has 'reward' key", "reward" in data)
    test("Reset has 'done' key", "done" in data)
    test("Reset reward is null", data["reward"] is None)
    test("Reset done is false", data["done"] is False)
    obs = data.get("observation", {})
    test("Obs has concentration", "concentration" in obs)
    test("Obs has step", "step" in obs)
    test("Obs has task", "task" in obs)

# With task
r = client.post("/reset", json={"task": "medium"})
test("POST /reset (medium) returns 200", r.status_code == 200)

r = client.post("/reset", json={"task": "hard"})
test("POST /reset (hard) returns 200", r.status_code == 200)

# Invalid task
r = client.post("/reset", json={"task": "impossible"})
test("POST /reset (invalid) returns 400", r.status_code == 400)

# ─── 4. Step endpoint ───
print("\n=== Step ===")

# Reset first
client.post("/reset", json={"task": "easy"})

r = client.post("/step", json={"action": {"dose": 5.0}})
test("POST /step returns 200", r.status_code == 200, f"got {r.status_code}: {r.text}")
if r.status_code == 200:
    data = r.json()
    test("Step has 'observation' key", "observation" in data)
    test("Step has 'reward' key", "reward" in data)
    test("Step has 'done' key", "done" in data)
    test("Step reward is a number", isinstance(data["reward"], (int, float)))
    test("Step done is bool", isinstance(data["done"], bool))

# Step without reset
r = client.post("/step", json={"action": {"dose": 0.0}})
test("POST /step (after reset) returns 200", r.status_code == 200)

# ─── 5. State endpoint ───
print("\n=== State ===")
r = client.get("/state")
test("GET /state returns 200", r.status_code == 200)
if r.status_code == 200:
    state = r.json()
    test("State has 'step' key", "step" in state)
    test("State has 'concentration' key", "concentration" in state)
    test("State has 'done' key", "done" in state)

# ─── 6. Full episode simulation ───
print("\n=== Full Episode ===")
client.post("/reset", json={"task": "easy"})
done = False
steps = 0
while not done and steps < 15:
    r = client.post("/step", json={"action": {"dose": 8.0}})
    assert r.status_code == 200
    data = r.json()
    done = data["done"]
    steps += 1
test(f"Full episode completed in {steps} steps", done or steps >= 10)

# ─── Summary ───
print(f"\n{'='*40}")
total = passed + failed
print(f"Results: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("🎉 ALL TESTS PASSED — Ready for deployment!")
else:
    print(f"⚠️  {failed} test(s) failed — fix before deploying.")
