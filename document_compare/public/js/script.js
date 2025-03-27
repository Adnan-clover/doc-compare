document.addEventListener("DOMContentLoaded", function() {
    console.log("redirected from Home")
    localStorage.removeItem("comparedData"); // Clears old response
    // console.log("Cache cleared before fetching new comparison data.");
  
    let compareData = JSON.parse(localStorage.getItem("comparedData"));
    // console.log(">>> compareData", compareData.section_details)
  
    if (compareData) {
        if (localStorage.getItem("uploadedData")) {
            localStorage.removeItem("uploadedData")
        }
        loadCompareData(compareData)
    }
  
    const uploadData = JSON.parse(localStorage.getItem("uploadedData"));
  
    if (uploadData) {
        document.getElementById("uploaded_files").style.display = "block";
        document.getElementById("output1").innerHTML = uploadData.file1;
        // console.log(">> file1",uploadData.file1);
        document.getElementById("output2").innerHTML = uploadData.file2;
        let filename1 = uploadData.filename1;
        let filename2 = uploadData.filename2;
        document.getElementById("original_name").innerHTML = filename1;
        document.getElementById("modified_name").innerHTML = filename2;
        let html1 = uploadData.file1;
        let html2 = uploadData.file2;
    }
  
    if (!compareData && !uploadData) {
        window.location.href = "/home_test";
    }
  
    document.getElementById("findDiffBtn").style.display = "block";
    // document.getElementById("resetBtn").style.display = "none";
    document.getElementById("uploaded_files").style.display = "block";
    // document.getElementById("reAnalyzeBtn").style.display = "none";
    document.getElementById("match-filter").style.display = "none";
    document.getElementById("icon-filter").style.display = "none";
  
    if (compareData) {
        document.getElementById("findDiffBtn").style.display = "none";
        // document.getElementById("reAnalyzeBtn").style.display = "block";
        document.getElementById("match-filter").style.display = "block";
        document.getElementById("icon-filter").style.display = "block";
    }
  
    let filename1 = "";
    let filename2 = "";
    let html1 = "";
    let html2 = "";
    let ollama_json = "";
    let last_highlight_el = null;
    let section_html1 = "";
    let section_html2 = "";
    let section_details = "";
    let document_id = null;
  
    document.getElementById("findDiffBtn").addEventListener("click", function() {
      console.log("findDiffBtn clicked");
    //   const csrf = document.querySelector('input[name="csrfmiddlewaretoken"]').value;
      const progressBar = document.getElementById("file");
  
      if (!uploadData.filename1 || !uploadData.filename2) {
          alert("Files are not properly uploaded. Please upload files first.");
          return;
      }
  
      // // Show progress bar and reset value
      // progressBar.style.display = "block";
      // progressBar.value = 0;
  
      // Show skeleton loaders
      showSkeletonLoaders();
  
      // Hide the button and show the loader
      document.getElementById("findDiffBtn").style.display = "none";
  
      // // Simulate progress increment (for visual effect)
      // let progressInterval = setInterval(() => {
      //     if (progressBar.value < 90) {
      //         progressBar.value += 10;
      //     }
      // }, 500);
  
      const compareData = {
          filename1: uploadData.filename1,
          filename2: uploadData.filename2,
          html1: uploadData.file1,
          html2: uploadData.file2,
      };
  
      fetch("/api/method/document_compare.api.load_sections", {
              method: "POST",
              headers: {
                  "Content-Type": "application/json",
                  "X-Frappe-CSRF-Token": frappe.csrf_token
              },
              body: JSON.stringify(compareData),
          })
          .then((response) => response.json())
          .then((data) => {
              let compareData = data.message
              if (compareData.success) {
                console.log("compare data successfull>>", compareData.message)
  
                  // // Stop progress at 100%
                  // clearInterval(progressInterval);
                  // progressBar.value = 100;
  
                  // Remove skeleton loaders
                  hideSkeletonLoaders();
  
                  // setTimeout(() => {
                  //     progressBar.style.display = "none";
                  // }, 500);
  
                  // Save compareData to localStorage
                  localStorage.setItem("comparedData", JSON.stringify(compareData));
                  section_html1 = compareData.file1;
                  console.log("section_html1", section_html1)
                  section_html2 = compareData.file2;
                  print("section_html2", section_html2)
                  section_details = compareData.section_details;
                  print("section_details", section_details)
                  document_id = compareData.document_id;
  
                  // Update document names
                  document.getElementById("original_name").innerHTML = compareData.original_doc_name;
                  document.getElementById("modified_name").innerHTML = compareData.modified_doc_name;
  
                  // Populate sections with real data
                  document.getElementById("output1").innerHTML = section_html1;
                  document.getElementById("output2").innerHTML = section_html2;
  
                  // Functions to process and display data
                  linkSections(section_details);
                  getOriginalSections(section_details);
                  totalCount(section_details);
                  applyFilter(section_details);
                  initializeLeaderLines(section_details, compareData);

                  console.log(">>>> line 144")
  
                  // Remove uploadedData from localStorage
                  if (localStorage.getItem("uploadedData")) {
                      localStorage.removeItem("uploadedData");
                  }
  
              } else {
                  alert("Comparison failed. Please try again.");
                  document.getElementById("findDiffBtn").style.display = "block";
                  hideSkeletonLoaders();
              }
          })
          .catch((error) => {
              alert("Error comparing files.");
              console.error(error);
              document.getElementById("findDiffBtn").style.display = "block";
              hideSkeletonLoaders();
          })
          .finally(() => {
              // Hide the loader and show buttons again
              document.getElementById("loader").style.display = "none";
            //   document.getElementById("reAnalyzeBtn").style.display = "block";
              document.getElementById("match-filter").style.display = "block";
          });
  });
  
  // Show skeleton loaders
  function showSkeletonLoaders() {
      const output1 = document.getElementById("output1");
      const output2 = document.getElementById("output2");
  
      output1.innerHTML = `
          <div class="skeleton skeleton-text"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-card"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-text"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-card"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-text"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-card"></div>
          <div class="skeleton skeleton-line"></div>
      `
      ;
  
      output2.innerHTML = `
          <div class="skeleton skeleton-text"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-card"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-text"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-card"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-text"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-card"></div>
          <div class="skeleton skeleton-line"></div>
      `;
  }
  
  // Remove skeleton loaders
  function hideSkeletonLoaders() {
      document.getElementById("output1").innerHTML = "";
      document.getElementById("output2").innerHTML = "";
  }
  
    document.querySelectorAll("#icon-filter .action-filter").forEach((icon) => {
        icon.addEventListener("click", function() {
            const selectedFilter = this.firstElementChild.classList.contains("fa-check") ?
                "matched" :
                this.firstElementChild.classList.contains("fa-times") ?
                "removed" :
                this.firstElementChild.classList.contains("fa-hourglass-half") ?
                "pending" :
                "all";
  
            // Apply fade effect instead of hiding
            applyFadeEffect(selectedFilter);
        });
    });
  
    // Initialize counts when page loads
    updateIconCounts();
  
  });
  
  
  function applyFilter(section_details) {
    document.getElementById("icon-filter").style.display = "flex";
    const checkboxes = document.querySelectorAll(".filter-checkbox");
    const nextBtns = document.querySelectorAll(".next-btn");
    const prevBtns = document.querySelectorAll(".prev-btn");
    const allCheckbox = document.querySelector(".filter-checkbox[value='all']");
    const products = document.querySelectorAll(".card_section");
  
    let currentIndex = {
      matched: 0,
      added: 0,
      removed: 0
      };
  
    // Ensure section_details is defined
    if (!section_details) {
        console.error("Error: section_details is undefined.");
        return;
    }
  
    // Update the counts next to checkboxes
    function updateCounts() {
        if (!Array.isArray(section_details)) {
            console.error("Error: section_details is not an array.");
            return;
        }
  
        let matchedCount = section_details.filter(item => item.change_type === "matched").length;
        let addedCount = section_details.filter(item => item.change_type === "added").length;
        let removedCount = section_details.filter(item => item.change_type === "removed").length;
  
        document.querySelector(".filter-checkbox[value='matched']").nextSibling.textContent = ` Matched (${matchedCount})`;
        document.querySelector(".filter-checkbox[value='added']").nextSibling.textContent = ` Added (${addedCount})`;
        document.querySelector(".filter-checkbox[value='removed']").nextSibling.textContent = ` Removed (${removedCount})`;
  
        // Show/hide sections
        products.forEach(product => {
          product.style.display = "block";
          });
  
          checkboxes.forEach(checkbox => {
              if (checkbox.checked) {
                  const value = checkbox.value;
                  products.forEach(product => {
                      if (product.classList.contains(value)) {
                          product.style.display = "block";
                      }
                  });
              }
          });
  
          // Navigate to next or previous section
          function navigateSections(type, direction) {
              const visibleSections = Array.from(document.querySelectorAll(`.card_section.${type}`))
                  .filter(section => section.style.display !== "none");
  
              if (visibleSections.length === 0) return;
  
              // Update index
              currentIndex[type] += direction === "next" ? 1 : -1;
  
              // Ensure index bounds
              if (currentIndex[type] < 0) currentIndex[type] = 0;
              if (currentIndex[type] >= visibleSections.length) currentIndex[type] = visibleSections.length - 1;
  
              // Scroll to section
              const targetSection = visibleSections[currentIndex[type]];
              targetSection.scrollIntoView({ behavior: "smooth", block: "center" });
  
              // Highlight current section
              visibleSections.forEach(section => section.classList.remove("highlight_a"));
              targetSection.classList.add("highlight_a");
  
              // Remove highlight after 5 seconds
              setTimeout(() => {
                  targetSection.classList.remove("highlight_a");
              }, 5000); // 5000ms = 5 seconds
          }
  
          // Event listeners for navigation buttons
          nextBtns.forEach(btn => {
              btn.addEventListener("click", () => {
                  const type = btn.getAttribute("data-type");
                  navigateSections(type, "next");
              });
          });
  
          prevBtns.forEach(btn => {
              btn.addEventListener("click", () => {
                  const type = btn.getAttribute("data-type");
                  navigateSections(type, "prev");
              });
          });
  
          // Checkbox change listener
          checkboxes.forEach(checkbox => {
              checkbox.addEventListener("change", () => {
                  applyFilter(window.section_details);
              });
          });
    }
  
  
    // grey fade in fade out function
    function filterProducts() {
        let selectedCategories = Array.from(checkboxes)
            .filter((checkbox) => checkbox.checked)
            .map((checkbox) => checkbox.value);
  
        products.forEach((product) => {
            let productCategory = product.getAttribute("data-category");
            console.log(productCategory, "pg");
            if (selectedCategories.includes("all") || selectedCategories.length === 0) {
                product.style.opacity = "1"; // Fully visible
                product.style.filter = "grayscale(0%)"; // No grayscale
            } else if (selectedCategories.includes(productCategory)) {
                product.style.opacity = "1"; // Fully visible
                product.style.filter = "grayscale(0%)";
            } else {
                product.style.opacity = "0.3"; // Faded effect
                product.style.filter = "grayscale(80%)"; // Slightly grey
            }
        });
    }
  
  
    checkboxes.forEach((checkbox) => {
        checkbox.addEventListener("change", function() {
            if (this.value === "all") {
                if (this.checked) {
                    //        alert("All click");
                    // If "All" is checked, uncheck all other filters
                    checkboxes.forEach((cb) => {
                        if (cb !== allCheckbox) cb.checked = false;
                    });
                }
            } else {
                // If any other checkbox is checked, uncheck "All"
                if (this.checked) {
                    allCheckbox.checked = false;
                }
            }
            filterProducts();
        });
    });
    updateCounts();
  };
  
  // The function linkSections takes an array of section details and sets up the linking
  function linkSections(section_details) {
    // Create mappings only if both IDs exist
    const mappingOriginalToModified = {};
    const mappingModifiedToOriginal = {};
  
    section_details.forEach((item) => {
        if (item.original_section_id && item.modified_section_id) {
            mappingOriginalToModified[item.original_section_id] =
                item.modified_section_id;
            mappingModifiedToOriginal[item.modified_section_id] =
                item.original_section_id;
        }
  
        // Assign border colors based on change_type
        const changeType = item.change_type.toLowerCase();
        let colorClass = "";
        if (changeType === "matched") colorClass = "matched";
        else if (changeType === "replaced") colorClass = "replaced";
        else if (changeType === "added") colorClass = "added";
        else if (changeType === "removed") colorClass = "removed";
  
        if (item.original_section_id) {
            const originalEl = document.getElementById(item.original_section_id);
            console.log(">>>> originalEl", originalEl)
            if (originalEl) {
                originalEl.classList.add(colorClass);
                originalEl.setAttribute("data-category", changeType);
            }
        }
        if (item.modified_section_id) {
            const modifiedEl = document.getElementById(item.modified_section_id);
            if (modifiedEl) {
                modifiedEl.classList.add(colorClass);
                modifiedEl.setAttribute("data-category", changeType);
            }
        }
    });
  
    section_html1 = document.getElementById("output1").innerHTML;
    section_html2 = document.getElementById("output2").innerHTML;
  
    document.addEventListener("click", function(event) {
        // Check if the clicked element is a section with the class 'card_section'
        if (event.target.closest(".card_section")) {
            const sectionId = event.target.closest(".card_section").id; // Get the id of the clicked section
            event.target
                .closest(".card_section")
                .scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });
            if (mappingOriginalToModified[sectionId]) {
                const targetId = mappingOriginalToModified[sectionId];
                const targetEl = document.getElementById(targetId);
                if (targetEl) {
                    targetEl.scrollIntoView({
                        behavior: "smooth",
                        block: "center"
                    });
                }
            }
        }
    });
  
    // Event delegation for modified sections
    document.addEventListener("click", function(event) {
        // Check if the clicked element is a section with the class 'card_section'
        if (event.target.closest(".card_section")) {
            const sectionId = event.target.closest(".card_section").id; // Get the id of the clicked section
            event.target
                .closest(".card_section")
                .scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });
            if (mappingModifiedToOriginal[sectionId]) {
                const targetId = mappingModifiedToOriginal[sectionId];
                const targetEl = document.getElementById(targetId);
                if (targetEl) {
                    targetEl.scrollIntoView({
                        behavior: "smooth",
                        block: "center"
                    });
                }
            }
        }
    });
  }
  
  // Function to update counts in the filter icons
  function updateIconCounts() {
    const checkCount = document.querySelectorAll('.three_icons .fa-check.active').length;
    const timesCount = document.querySelectorAll('.three_icons .fa-times.active').length;
    const hourglassCount = document.querySelectorAll('.three_icons .fa-hourglass-half.active').length;
  
    // Update the counts inside the icon-filter div
    document.querySelector("#icon-filter .fa-check + span").textContent = checkCount;
    document.querySelector("#icon-filter .fa-times + span").textContent = timesCount;
    document.querySelector("#icon-filter .fa-hourglass-half + span").textContent = hourglassCount;
  }
  
  // Function to apply fade effect instead of hiding sections
  function applyFadeEffect(filterType) {
    const sections = document.querySelectorAll(".card_section");
  
    sections.forEach((section) => {
        const isChecked = section.querySelector(".fa-check.active");
        const isRejected = section.querySelector(".fa-times.active");
        const isPending = section.querySelector(".fa-hourglass-half.active");
  
        if (
            (filterType === "matched" && isChecked) ||
            (filterType === "removed" && isRejected) ||
            (filterType === "pending" && isPending) ||
            (filterType === "all")
        ) {
            section.style.opacity = "1"; // Fully visible
            section.style.filter = "grayscale(0%)"; // Normal color
        } else {
            section.style.opacity = "0.3"; // Faded effect
            section.style.filter = "grayscale(80%)"; // Slightly grey
        }
    });
  }
  
  
  function totalCount(section_details) {
    // You can modify this to calculate a more specific count if needed
    const totalCount = section_details ? section_details.length : 0;
    // Update the count next to the 'eye' icon
    document.querySelector("#icon-filter .fa-layer-group + span").textContent = totalCount;
  }
  
  // Updated Code //
  function highlightIcon(selectedIcon) {
    const iconGroup = selectedIcon.closest('.icon-group'); // Get the icon container
    const icons = iconGroup.querySelectorAll('i'); // Get all icons in the same row
  
    // Reset all icons in the same group
    icons.forEach(icon => {
        icon.classList.remove('active', 'faded');
        icon.style.color = ''; // Reset color
        icon.style.opacity = "0.7"; // Fully visible
        icon.style.pointerEvents = "auto"; // Ensure all buttons are clickable
    });
  
    // Highlight the clicked icon
    selectedIcon.classList.add('active');
  
    // Set color dynamically
    if (selectedIcon.classList.contains('fa-check')) {
        selectedIcon.style.color = '#28a745'; // Green
    } else if (selectedIcon.classList.contains('fa-times')) {
        selectedIcon.style.color = '#df4545'; // Red
    } else if (selectedIcon.classList.contains('fa-hourglass-half')) {
        selectedIcon.style.color = '#ffb800'; // Yellow
    }
  
    // Fade out other icons but keep them clickable
    icons.forEach(icon => {
        if (icon == selectedIcon) {
            icon.classList.add('faded');
            icon.style.opacity = "1"; // Slight fade but still clickable
            icon.style.pointerEvents = "auto"; // Keep clickable
        }
    });
  
    updateIconCounts();
  }
  
  
  function getOriginalSections(data) {
    const result = data.map(item => {
        if (!item.original_section_id) return null;
        if (!item.modified_section_id) return null;
  
        const element = document.getElementById(item.original_section_id);
        const element2 = document.getElementById(item.modified_section_id);
        if (!element || !element2) {
            console.warn(`No element found for ID ${item.original_section_id}`);
            return null;
        }
  
        // Prevent duplicate .three_icons elements
        if (element.querySelector('.three_icons')) {
            console.log(`.three_icons already exists for ${item.original_section_id}, skipping.`);
            return {
                id: item.original_section_id,
                element
            };
        }
  
        const pageNumber = element.getAttribute("page-number");
        const pageNumber2 = element2.getAttribute("page-number");
        // console.log('pno', pageNumber);
        // console.log('pno2', pageNumber2);
  
        // Create the icons container
        const threeIcons = document.createElement('div');
        threeIcons.className = 'three_icons';
  
        const iconGroup = document.createElement('div');
        iconGroup.className = 'icon-group';
        iconGroup.innerHTML = `
              <i class="fas fa-check" onclick="highlightIcon(this); markSection(this);"></i>
              <i class="fas fa-times" onclick="highlightIcon(this); markSection(this);"></i>
              <i class="fas fa-hourglass-half" onclick="highlightIcon(this); markSection(this);"></i>
          `;
  
        const pageNo = document.createElement('div');
        pageNo.className = 'pageNo';
        pageNo.innerHTML = `<span>Page ${pageNumber}</span>`;
  
        const page2No = document.createElement('div');
        page2No.className = 'page2No';
        page2No.innerHTML = `<span>Page ${pageNumber2}</span>`;
  
        // Append elements
        threeIcons.appendChild(iconGroup);
        threeIcons.appendChild(pageNo);
        //const pageNo2 = document.createElement('div');
        //pageNo2.className = 'pageIndex';
        //const pageIndex = document.querySelector('.pageIndex');
        //if (pageIndex) pageIndex.appendChild(page2No);
  
        element.appendChild(threeIcons);
        element2.appendChild(page2No);
        return {
            id: item.original_section_id,
            element
        };
    }).filter(entry => entry !== null);
  
    return result;
  }
  
  
  function initializeLeaderLines(section_details, compareData) {
    if (typeof AnimEvent === 'undefined') {
        window.AnimEvent = {
            add: function(callback) {
                return callback;
            }
        };
    }
  
    let leaderLines1 = [];
    let leaderLines = [];
  
    leaderLines1.forEach(line1 => line1.remove());
    leaderLines1 = [];
  
    leaderLines.forEach(line => line.remove());
    leaderLines = [];
  
    const matchedData = section_details.filter(
        item => item.change_type === "matched" && item.original_section_id && item.modified_section_id
    );
  
    matchedData.forEach(item => {
        const originalEl = document.getElementById(item.original_section_id);
        const modifiedEl = document.getElementById(item.modified_section_id);
  
        if (originalEl && modifiedEl) {
            let line1 = new LeaderLine(
                LeaderLine.mouseHoverAnchor(modifiedEl),
                originalEl, {
                    color: "green",
                    endPlug: "arrow",
                    startPlug: "arrow1",
                    path: 'fluid',
                    dash: {
                        animation: true
                    },
                    startSocket: "left",
                    endSocket: "right"
                }
            );
            let line = new LeaderLine(
                LeaderLine.mouseHoverAnchor(originalEl),
                modifiedEl, {
                    color: "green",
                    endPlug: "arrow",
                    startPlug: "arrow1",
                    path: 'fluid',
                    dash: {
                        animation: true
                    },
                    startSocket: "right",
                    endSocket: "left"
                }
            );
            leaderLines.push(line);
            leaderLines1.push(line1);
        }
    });
    console.log("Lines created:", leaderLines.length);
  
    const output1 = document.getElementById("box1");
    const output2 = document.getElementById("box2");
  
    function repositionLines() {
        leaderLines.forEach(line => line.position());
        leaderLines1.forEach(line1 => line1.position());
    }
  
    output1.addEventListener('scroll', AnimEvent.add(repositionLines), false);
    output2.addEventListener('scroll', AnimEvent.add(repositionLines), false);
  
    document.querySelector(".filter-checkbox[value='matched']").nextSibling.textContent = ` Matched (${compareData.matched_count})`;
    document.querySelector(".filter-checkbox[value='added']").nextSibling.textContent = ` Added (${compareData.added_count})`;
    document.querySelector(".filter-checkbox[value='removed']").nextSibling.textContent = ` Removed (${compareData.removed_count})`;
  }
  
  
  function markSection(selectedIcon) {
    // const csrf_token = document.querySelector('input[name="csrfmiddlewaretoken"]').value;
    const iconGroup = selectedIcon.closest('.icon-group'); // Get the icon container
    const icons = iconGroup.querySelectorAll('i'); // Get all icons in the same row
    let section = selectedIcon.closest("section");
  
    if (section) {
  
        let action = ''; // Initialize action variable
  
        if (selectedIcon.classList.contains('fa-check')) {
            action = 'approved';
        } else if (selectedIcon.classList.contains('fa-times')) {
            action = 'rejected';
        } else if (selectedIcon.classList.contains('fa-hourglass-half')) {
            action = 'pending';
        }
        const compareData = JSON.parse(localStorage.getItem("comparedData"));
        document_id = compareData.document_id;
  
        const markData = {
            document_id: document_id, // Example document ID
            section_id: section.id,
            action: action,
        };
  
        // Sending the POST request
        fetch("/api/method/document_compare.api.mark_sections", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Frappe-CSRF-Token": frappe.csrf_token
                },
                body: JSON.stringify(markData),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json(); // Parse JSON response
            })
            .then(data => {
                // Ensure section_info is an object, avoiding unnecessary JSON parsing
                let section_info = (typeof data.section_info === "string") ?
                    JSON.parse(data.section_info) :
                    data.section_info || {};
  
                // Retrieve existing comparedData from localStorage or initialize an empty object
                let compared_data = JSON.parse(localStorage.getItem("comparedData")) || {};
  
                // Ensure section_info exists in compared_data
                compared_data.section_info = compared_data.section_info || {};
  
                // Merge new section_info into the existing section_info
                Object.assign(compared_data.section_info, section_info);
  
                // Store the updated comparedData back to localStorage
                localStorage.setItem("comparedData", JSON.stringify(compared_data));
  
                //                console.log("Updated comparedData:", compared_data);
            })
            .catch(error => {
                console.error("Error:", error); // Handle error
                // Optionally show error message to user
            });
    }
  };
  
  
  function loadCompareData(compareData) {
    console.log('compareData---', compareData)
    section_html1 = compareData.file1;
    section_html2 = compareData.file2;
    section_details = compareData.section_details;
    document_id = compareData.document_id;
  
    document.getElementById("original_name").innerHTML = compareData.original_doc_name;
    document.getElementById("modified_name").innerHTML = compareData.modified_doc_name;
  
    document.getElementById("output1").innerHTML = section_html1;
    document.getElementById("output2").innerHTML = section_html2;
  
    linkSections(section_details);
    getOriginalSections(section_details);
    totalCount(section_details); // Update the total count
    applyFilter(section_details);
    initializeLeaderLines(section_details, compareData);
  
    if (localStorage.getItem("comparedData")) {
        let comparedData = JSON.parse(localStorage.getItem("comparedData"));
        let sectionInfo = comparedData.section_info;
  
        if (sectionInfo) {
            let status_list = {
                "approved": "fas fa-check",
                "rejected": "fas fa-times",
                "pending": "fas fa-hourglass-half"
            };
  
            for (let sectionId in sectionInfo) {
                let section = document.getElementById(sectionId);
  
                if (section) {
                    let status = sectionInfo[sectionId];
                    let iconClass = status_list[status];
  
                    if (iconClass) {
                        let icon = section.querySelector(`i.${iconClass.split(" ").join(".")}`);
  
                        if (icon) {
                            highlightIcon(icon);
                        }
                    }
                }
            }
        }
  
    } else {
        window.location.href = "/home_test";
    }
  };
  
  
  function reAnalyzeCompareData() {
  
    console.log("reAnalyze clicked");
    const csrf = document.querySelector('input[name="csrfmiddlewaretoken"]').value;
    const progressBar = document.getElementById("file");
  
    if (!localStorage.getItem("comparedData")) {
        alert("Files are not properly uploaded. Please upload files first.");
        return;
    }
  
    let compareData = JSON.parse(localStorage.getItem("comparedData"));
  
  //   // Show progress bar and reset value
  //   progressBar.style.display = "block";
  //   progressBar.value = 0;
  
    // Show skeleton loaders
    showSkeletonLoaders();
  
    // Hide the button and show the loader
    document.getElementById("reAnalyzeBtn").style.display = "none";
  
  //   // Simulate progress increment (for visual effect, you can remove this if using real-time updates)
  //   let progressInterval = setInterval(() => {
  //       if (progressBar.value < 90) {
  //           progressBar.value += 10;
  //       }
  //   }, 500);
  
    const compareJsonData = {
        filename1: compareData.original_doc_name,
        filename2: compareData.modified_doc_name,
        html1: compareData.file1,
        html2: compareData.file2,
        section_details: compareData.section_details,
        document_id : compareData.document_id
    };
  
    fetch("/sections/reanalyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrf
            },
            body: JSON.stringify(compareJsonData),
        })
        .then((response) => response.json())
        .then((compareData) => {
            if (compareData.success) {
  
              //   // Stop progress at 100% when done
              //   clearInterval(progressInterval);
              //   progressBar.value = 100;
  
                // Remove skeleton loaders
                hideSkeletonLoaders();
  
              //   // Simulate slight delay before hiding progress bar
              //   setTimeout(() => {
              //       progressBar.style.display = "none";
              //   }, 500);
  
                // Save compareData to localStorage
                localStorage.setItem("comparedData", JSON.stringify(compareData));
                section_html1 = compareData.file1;
                section_html2 = compareData.file2;
                section_details = compareData.section_details;
                document_id = compareData.document_id;
  
                document.getElementById("original_name").innerHTML = compareData.original_doc_name;
                document.getElementById("modified_name").innerHTML = compareData.modified_doc_name;
  
                document.getElementById("output1").innerHTML = section_html1;
                document.getElementById("output2").innerHTML = section_html2;
  
                linkSections(section_details);
                getOriginalSections(section_details);
                totalCount(section_details); // Update the total count
                applyFilter(section_details);
                initializeLeaderLines(section_details, compareData);
  
                if (localStorage.getItem("uploadedData")) {
                    localStorage.removeItem("uploadedData");
                }
  
            } else {
                alert("Comparison failed. Please try again.");
                document.getElementById("findDiffBtn").style.display = "block";
                hideSkeletonLoaders();
            }
        })
        .catch((error) => {
            alert("Error comparing files.");
            console.error(error);
            document.getElementById("findDiffBtn").style.display = "block";
            hideSkeletonLoaders();
        })
        .finally(() => {
            // Hide the loader and show the button again
            document.getElementById("loader").style.display = "none";
            document.getElementById("findDiffBtn").style.display = "none";
            // document.getElementById("resetBtn").style.display = "none";
            document.getElementById("reAnalyzeBtn").style.display = "block";
            document.getElementById("match-filter").style.display = "block";
        });
  };
  
  // Show skeleton loaders
  function showSkeletonLoaders() {
      const output1 = document.getElementById("output1");
      const output2 = document.getElementById("output2");
  
      output1.innerHTML = `
          <div class="skeleton skeleton-text"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-card"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-text"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-card"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-text"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-card"></div>
          <div class="skeleton skeleton-line"></div>
      `
      ;
  
      output2.innerHTML = `
          <div class="skeleton skeleton-text"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-card"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-text"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-card"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-text"></div>
          <div class="skeleton skeleton-line"></div>
          <div class="skeleton skeleton-card"></div>
          <div class="skeleton skeleton-line"></div>
      `;
  }
  
  // Remove skeleton loaders
  function hideSkeletonLoaders() {
      document.getElementById("output1").innerHTML = "";
      document.getElementById("output2").innerHTML = "";
  }
  
  