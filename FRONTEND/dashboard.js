const uploadimg = document.querySelector(".uploadimg");
const uploadimgc = document.querySelector(".uploadimgc");
const input1 = document.querySelector("#input1");
const img1 = document.querySelector(".img1");
const uploadbtn = document.querySelector(".uploadbtn");
const firstb1 = document.querySelector(".firstb1");
const results = document.querySelector(".results");

results.addEventListener("click",()=>{
    everythingcontainer1.style.display = "block";
})

firstb1.addEventListener("click",()=>{
    Everything.style.display = "flex";
})



input1.addEventListener("change",uploadimginp);
function uploadimginp(){
    let imglink = URL.createObjectURL(input1.files[0]);
    img1.src = `${imglink}`;
    // uploadimg.style.backgroundImage = `url(${imglink})`;
    img1.style.display = "block";
    // uploadimg.style.display = "none";
    uploadimgc.style.display = "none";
    uploadimg.style.backgroundColor= "rgb(228, 228, 228)";
}


uploadbtn.addEventListener("click",async()=>{
    try{const input = document.querySelector("#input1").files[0];
    if(!input) return;
    uploadbtn.innerText = "Predicting...";

    let formData = new FormData();
    formData.append("data",input);

    const result = await fetch("http://127.0.0.1:8000/braintumor",{
        method:"POST",
        body:formData
    });

    const data = await result.json();
    const answer = document.querySelector(".answer");
    const answer1 = document.querySelector(".answer1");

    answer.style.display = "block";
    answer1.style.display = "block";

    answer.innerText = `The Animal present in image is ${data.pred_label}.`
    answer1.innerText = `Model confidence score is ${data.conf.toFixed(3)}.`
    uploadbtn.innerText = "Upload Image";
    }
    catch(error){
        alert("Internal server error.");
        return;
    }
})

uploadimg.addEventListener("dragover",(e)=>{
    e.preventDefault();
    uploadimg.style.backgroundColor = "rgb(205, 205, 205)";
})
uploadimg.addEventListener("drop",(e)=>{
    e.preventDefault();
    input1.files = e.dataTransfer.files;
    uploadimginp();
})



const projectname = document.querySelector(".projectname");
const Everything = document.querySelector(".Everything");

projectname.addEventListener("click",()=>{
    Everything.style.display = "none";
});


const projectname1 = document.querySelector(".projectname1");
const everythingcontainer1 = document.querySelector(".everythingcontainer1");

projectname1.addEventListener("click",()=>{
    everythingcontainer1.style.display = "none";
})