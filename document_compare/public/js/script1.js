document.addEventListener("DOMContentLoaded", function() {
    console.log("Frappe-integrated script loaded");

    // progress bar with 100% completion.
    // document.getElementById("upload_btn").addEventListener("click", function() {
    //     console.log("Upload button clicked");

    //     // Get file inputs and other elements
    //     const file1 = document.getElementById("fileInput1").files[0];
    //     const file2 = document.getElementById("fileInput2").files[0];
    //     // const csrf = document.querySelector('input[name="csrfmiddlewaretoken"]').value;
    //     const csrfToken = "{{ frappe.session.csrf_token }}";
    //     const progressBar = document.getElementById("progressBar");
    //     const progressContainer = document.getElementById("progressContainer");

    //     // Validation
    //     if (!file1 || !file2) {
    //         alert("Please select both files before uploading.");
    //         return;
    //     }

    //     if (
    //         file1.name.split(".").pop().toLowerCase() !== "docx" ||
    //         file2.name.split(".").pop().toLowerCase() !== "docx"
    //     ) {
    //         alert("Invalid file extension. Please upload .docx files.");
    //         return;
    //     }

    //     const maxSize = 2 * 1024 * 1024;
    //     if (file1.size > maxSize || file2.size > maxSize) {
    //         alert("Each file must be smaller than 2MB.");
    //         return;
    //     }

    //     // Show progress bar
    //     progressContainer.style.display = "block";
    //     progressBar.style.width = "0%";
    //     progressBar.textContent = "0%";

    //     // Prepare form data
    //     const formData = new FormData();
    //     formData.append("file1", file1);
    //     formData.append("file2", file2);

    //     // XMLHttpRequest setup
    //     const xhr = new XMLHttpRequest();
    //     xhr.open("POST", "/upload", true);
    //     xhr.setRequestHeader("X-CSRFToken", csrfToken);

    //     // Track upload progress (0 to 85%)
    //     xhr.upload.addEventListener("progress", function(e) {
    //         if (e.lengthComputable) {
    //             let percentComplete = Math.round((e.loaded / e.total) * 95);
    //             incrementProgressTo(percentComplete);
    //         }
    //     });

    //     // Handle upload completion
    //     xhr.onload = function() {
    //         if (xhr.status === 200) {
    //             const response = JSON.parse(xhr.responseText);
    //             if (response.success) {
    //                 incrementProgressTo(100); // Successful response, move to 100%
    //                 setTimeout(() => {

    //                     if (localStorage.getItem("comparedData")) {
    //                         localStorage.removeItem("comparedData");
    //                     }

    //                     localStorage.setItem("uploadedData", JSON.stringify(response));
    //                     window.location.href = "/compare"; // Redirect after success
    //                 }, 1000);
    //             } else {
    //                 alert(response.message || "Upload failed. Please try again.");
    //                 progressContainer.style.display = "none";
    //             }
    //         } else {
    //             alert("Server error: " + xhr.responseText);
    //             progressContainer.style.display = "none";
    //         }
    //     };

    //     // Network error handler
    //     xhr.onerror = function() {
    //         alert("Network error. Please try again.");
    //         progressContainer.style.display = "none";
    //     };

    //     // Send form data
    //     xhr.send(formData);

    //     // Progress bar logic
    //     function updateProgressBar(percent) {
    //         progressBar.style.width = percent + "%";
    //         progressBar.textContent = percent + "%";
    //     }

    //     function incrementProgressTo(target) {
    //         let currentProgress = parseInt(progressBar.textContent);
    //         const increment = currentProgress >= 85 ? 1 : 1; // 2 until 85%, then 1

    //         const interval = setInterval(() => {
    //             if (currentProgress < target) {
    //                 currentProgress += increment;
    //                 if (currentProgress > target) currentProgress = target; // Avoid overshooting
    //                 updateProgressBar(currentProgress);
    //             } else {
    //                 clearInterval(interval);
    //                 if (target === 100) {
    //                     progressBar.textContent = "Upload Complete!";
    //                 }
    //             }
    //         }, 350); // Adjust the speed of the progress
    //     }

    //     function holdProgressAt(percent) {
    //         updateProgressBar(percent);
    //     }
    // });


    // Handle file uploads and progress bar
    document.getElementById("upload_btn").addEventListener("click", function() {
        console.log("Upload button clicked");
    
        const file1 = document.getElementById("fileInput1").files[0];
        const file2 = document.getElementById("fileInput2").files[0];
    
        if (!file1 || !file2) {
            frappe.msgprint("Please select both files before uploading.");
            return;
        }
    
        if (
            file1.name.split(".").pop().toLowerCase() !== "docx" ||
            file2.name.split(".").pop().toLowerCase() !== "docx"
        ) {
            frappe.msgprint("Invalid file extension. Please upload .docx files.");
            return;
        }
    
        const maxSize = 2 * 1024 * 1024; // 2MB limit
        if (file1.size > maxSize || file2.size > maxSize) {
            frappe.msgprint("Each file must be smaller than 2MB.");
            return;
        }
    
        const progressBar = document.getElementById("progressBar");
        const progressContainer = document.getElementById("progressContainer");
        progressContainer.style.display = "block";
        updateProgressBar(0);
    
        const formData = new FormData();
        formData.append("file1", file1);
        formData.append("file2", file2);
    
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/method/document_compare.api.upload_files", true);
        xhr.setRequestHeader("Accept", "application/json");
        xhr.setRequestHeader("X-Frappe-CSRF-Token", frappe.csrf_token);
    
        xhr.upload.onprogress = function(e) {
            if (e.lengthComputable) {
                const percentComplete = Math.round((e.loaded / e.total) * 100);
                updateProgressBar(percentComplete);
            }
        };
    
        xhr.onload = function() {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                if (response.message && response.message.success) {
                    incrementProgressTo(100);
                    frappe.msgprint("Files uploaded and compared successfully!");
    
                    localStorage.setItem("uploadedData", JSON.stringify(response.message));
                    setTimeout(() => {
                        window.location.href = "/compare";
                    }, 1000);
                } else {
                    frappe.msgprint(response.message.message || "Failed to compare documents.");
                    progressContainer.style.display = "none";
                }
            } else {
                frappe.msgprint("Server error. Please try again.");
                progressContainer.style.display = "none";
            }
        };
    
        xhr.onerror = function() {
            frappe.msgprint("Upload failed. Please check your internet connection.");
            progressContainer.style.display = "none";
        };
    
        xhr.send(formData);
    
        function updateProgressBar(percent) {
            progressBar.style.width = percent + "%";
            progressBar.textContent = percent + "%";
        }
    
        function incrementProgressTo(target) {
            let currentProgress = parseInt(progressBar.textContent) || 0;
            const increment = currentProgress >= 85 ? 1 : 2;
    
            const interval = setInterval(() => {
                if (currentProgress < target) {
                    currentProgress += increment;
                    if (currentProgress > target) currentProgress = target;
                    updateProgressBar(currentProgress);
                } else {
                    clearInterval(interval);
                    if (target === 100) {
                        progressBar.textContent = "Upload Complete!";
                    }
                }
            }, 350);
        }
    });
    
});

function viewEditDocument(documentId) {
    const csrf_token = document.querySelector('input[name="csrfmiddlewaretoken"]').value; // Get CSRF token

    const requestData = {
        document_id: documentId
    };

    fetch("/document/view", { // Change this URL to your backend endpoint
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrf_token
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            console.log("Response:", data);
            localStorage.setItem("comparedData", JSON.stringify(data));
            window.location.href = "/compare";
        })
        .catch(error => {
            console.error("Error:", error);
        });
}
