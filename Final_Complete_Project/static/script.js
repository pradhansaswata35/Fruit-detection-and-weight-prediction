// scrolling effect for navbar
function scrollToSection(id) {
  const element = document.getElementById(id);
  const offset = window.innerHeight * 0.10; // 10vh
  const bodyRect = document.body.getBoundingClientRect().top;
  const elementRect = element.getBoundingClientRect().top;
  const position = elementRect - bodyRect - offset;

  window.scrollTo({
    top: position,
    behavior: 'smooth'
  });

  // Set active class
  const navbarButtons = document.querySelectorAll('.navbar button');
  navbarButtons.forEach(btn => btn.classList.remove('active')); // remove from all
  event.target.classList.add('active'); // add to clicked one
}


document.addEventListener("DOMContentLoaded", function () {
  const uploadBtn = document.getElementById("uploadBtn");
  const captureBtn = document.getElementById("captureBtn");
  const giveInputBtn = document.getElementById("giveInputBtn");

  let selectedMode = ""; // "upload" or "capture"

  // function for make changes in website table
  function updatePredictionTable(data) {
    const tbody = document.querySelector("#predictionTable tbody");
    const totalRow = document.getElementById("totalRow");

    // Clear old rows except total row
    while (tbody.firstChild && tbody.firstChild !== totalRow) {
      tbody.removeChild(tbody.firstChild);
    }

    // Add new rows
    data.data.forEach((item, index) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${index + 1}</td>
        <td>${item.class}</td>
        <td>${item.height}</td>
        <td>${item.width}</td>
        <td>${item.color}</td>
        <td>${item.weight} g</td>
      `;
      tbody.insertBefore(row, totalRow);
    });

    // Update total weight
    totalRow.querySelector("td:last-child").innerHTML = `<strong>${data.total_weight} g</strong>`;
  }


  // Toggle active state and set mode
  function setActiveButton(button) {
    uploadBtn.classList.remove("active");
    captureBtn.classList.remove("active");

    button.classList.add("active");

    if (button === uploadBtn) {
      selectedMode = "upload";
    } else if (button === captureBtn) {
      selectedMode = "capture";
    }
  }

  // Add event listeners
  uploadBtn.addEventListener("click", () => setActiveButton(uploadBtn));
  captureBtn.addEventListener("click", () => setActiveButton(captureBtn));

  // Give Input button functionality
  giveInputBtn.addEventListener("click", () => {
    if (!selectedMode) {
      alert("Please select Upload Image or Capture Images first.");
      return;
    }


    if (selectedMode === "upload") {
      // Call backend for upload image action
      fetch("/upload-handler", {
        method: "POST",
      })
        .then((res) => res.json())
        .then((data) => {
          console.log("Upload response:", data);
          // Here update your table if needed

          updatePredictionTable(data);

          function updatePredictionTable(data) {
            const tbody = document.querySelector("#predictionTable tbody");
            const totalRow = document.getElementById("totalRow");

            // Clear old rows except total row
            while (tbody.firstChild && tbody.firstChild !== totalRow) {
              tbody.removeChild(tbody.firstChild);
            }

            // Add new rows
            data.data.forEach((item, index) => {
              const row = document.createElement("tr");
              row.innerHTML = `
                <td>${index + 1}</td>
                <td>${item.class}</td>
                <td>${item.height}</td>
                <td>${item.width}</td>
                <td>${item.color}</td>
                <td>${item.weight} g</td>
              `;
              tbody.insertBefore(row, totalRow);
            });

            // Update total weight
            totalRow.querySelector("td:last-child").innerHTML = `<strong>${data.total_weight} g</strong>`;
          }


        })
        .catch((err) => {
          console.error("Upload error:", err);
        });


    } else if (selectedMode === "capture") {
      // Call backend for capture image action
      fetch("/capture-handler", {
        method: "POST",
      })
        .then((res) => res.json())
        .then((data) => {
          console.log("Capture response:", data);
          // Here update your table if needed

          updatePredictionTable(data);

          function updatePredictionTable(data) {
            const tbody = document.querySelector("#predictionTable tbody");
            const totalRow = document.getElementById("totalRow");

            // Clear old rows except total row
            while (tbody.firstChild && tbody.firstChild !== totalRow) {
              tbody.removeChild(tbody.firstChild);
            }

            // Add new rows
            data.data.forEach((item, index) => {
              const row = document.createElement("tr");
              row.innerHTML = `
                <td>${index + 1}</td>
                <td>${item.class}</td>
                <td>${item.height}</td>
                <td>${item.width}</td>
                <td>${item.color}</td>
                <td>${item.weight} g</td>
              `;
              tbody.insertBefore(row, totalRow);
            });

            // Update total weight
            totalRow.querySelector("td:last-child").innerHTML = `<strong>${data.total_weight} g</strong>`;
          }

        })
        .catch((err) => {
          console.error("Capture error:", err);
        });
    }
  });
});


