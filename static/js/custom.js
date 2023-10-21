$('.run').click(function(){
    var code = editor.getSession().getValue();
    var language = $("#languages").val();
    $.ajax({
        type: "GET",
        url: "/user-code",
        contentType: 'application/json;charset=UTF-8',
        data: {'code':code, 'language':language},
        success: function(data,status){
            document.getElementById('show-term').style.display = 'block';
            document.getElementById('show-term-label').style.display = 'block';
            document.getElementById('hide-term').style.display = 'block';
            var output = JSON.parse(data);
            console.log(output)
            $('#show-term').text(output);
        }
    });
});

$('.hear').click(function(){
    var tts = editor.getValue();
    $.ajax({
        type: "GET",
        url: "/text-to-speech",
        contentType: 'application/json;charset=UTF-8',
        data: {'tts':tts},
        success: function(data,status){

        }
    });
});

$('.speak').click(function(){
    $.ajax({
        type: "GET",
        url: "/speech-to-text",
        contentType: 'application/json;charset=UTF-8',
        data: {},
        success: function(data,status){
            var visual = JSON.parse(data);
            console.log(visual);
            editor.setValue(visual);
        }
    });
});

$('.click-es').click(function(){
    document.getElementById('show-es').style.display = 'block';
    document.getElementById('show-es-label').style.display = 'block';
    document.getElementById('hide-es').style.display = 'block';
});

$('.rem-es').click(function(){
    $('#hide-es').hide();
    $('#show-es').hide();
    $('#show-es-label').hide();
});

$('.click-suggest').click(function(){
    document.getElementById('show-suggest').style.display = 'block';
    document.getElementById('show-suggest-label').style.display = 'block';
    document.getElementById('hide-suggest').style.display = 'block';
});

$('.rem-suggest').click(function(){
    $('#hide-suggest').hide();
    $('#show-suggest').hide();
    $('#show-suggest-label').hide();
});

$('.click-term').click(function(){
    document.getElementById('show-term').style.display = 'block';
    document.getElementById('show-term-label').style.display = 'block';
    document.getElementById('hide-term').style.display = 'block';
});

$('.rem-term').click(function(){
    $('#hide-term').hide();
    $('#show-term').hide();
    $('#show-term-label').hide();
});

$('.click-pa').click(function(){
    document.getElementById('show-pa').style.display = 'block';
    document.getElementById('show-pa-label').style.display = 'block';
    document.getElementById('hide-pa').style.display = 'block';
});

$('.rem-pa').click(function(){
    $('#hide-pa').hide();
    $('#show-pa').hide();
    $('#show-pa-label').hide();
});

$('.click-pp').click(function(){
    document.getElementById('show-pp').style.display = 'block';
    document.getElementById('show-pp-label').style.display = 'block';
    document.getElementById('hide-pp').style.display = 'block';
});

$('.rem-pp').click(function(){
    $('#hide-pp').hide();
    $('#show-pp').hide();
    $('#show-pp-label').hide();
});

$('.click-pc').click(function(){
    document.getElementById('show-pc').style.display = 'block';
    document.getElementById('show-pc-label').style.display = 'block';
    document.getElementById('hide-pc').style.display = 'block';
});

$('.rem-pc').click(function(){
    $('#hide-pc').hide();
    $('#show-pc').hide();
    $('#show-pc-label').hide();
});

$('.click-vizout').click(function(){
    document.getElementById('show-vizout').style.display = 'block';
    document.getElementById('show-vizout-label').style.display = 'block';
    document.getElementById('show-vizout-min').style.display = 'block';
});

$('.close-vizout').click(function(){
    $('#show-vizout-min').hide();
    $('#show-vizout-label').hide();
    $('#show-vizout').hide();
});

let editor;

window.onload = function(){
    editor = ace.edit("editor");
    editor.setTheme("ace/theme/monokai");
}

function changeLanguage(){
    let language = $("#languages").val();
    if (language == 'python')editor.session.setMode("ace/mode/python");
}