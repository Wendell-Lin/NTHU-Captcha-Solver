myFunction();
async function myFunction(){
    var text = document.getElementById("captcha_code");
    var captcha = "";
    captcha = (await getCaptcha());
    console.log(captcha)
    text.value = captcha;
}

async function getCaptcha(){
    var cimg = document.getElementById("captcha_image").src;
    var ccookie = await getCookie('PHPSESSID');
    var ret = ""
    ret = await sendRequest(cimg, ccookie);
    return ret;
}

function getCookie(cname) {
    var name = cname + "=";
    var ca = document.cookie.split(';');
    for(var i=0; i<ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0)==' ') c = c.substring(1);
        if (c.indexOf(name) == 0) return c.substring(name.length,c.length);
    }
    return "";
}

async function sendRequest(_url, _cookie){
    const today = new Date();
    const h = today.getUTCHours();
    var url = "";
    if(h < 12) url = "https://eeclass-captcha-server.herokuapp.com/captcha_predict"
    else url = "https://elearn-captcha-server.herokuapp.com/captcha_predict"
    var data = {"url":  _url,
                "cookie": _cookie};
    var result = await fetch(url, {
        method:'POST',
        body: JSON.stringify(data),
        headers: {
            'content-type': 'application/json'
        }
    });
    var res = await result.json();
    var ret = ""
    ret = res.result;
    return ret; 
}