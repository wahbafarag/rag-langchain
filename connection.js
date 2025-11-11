try {
  const response = await fetch("http://localhost:1234/v1/models");
  const data = await response.json();
  console.log("LM Studio is running! Available models:");
  console.log(data.data.map((m) => m.id));
} catch (error) {
  console.error("Cannot connect to LM Studio on port 1234");
  console.error("Error:", error.message);
}
