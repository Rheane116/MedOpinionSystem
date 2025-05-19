// 绑定按钮点击事件
document.getElementById('downloadButton').addEventListener('click', function () {
    // 向服务器请求下载URL
    fetch('/page/get-download-url') // 替换为你的后端API地址
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.url) {
                // 创建一个隐藏的<a>标签，触发下载
                const link = document.createElement('a');
                link.href = data.url;
                link.download = 'filename.ext'; // 设置下载文件的默认名称
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            } else {
                alert('Failed to get download URL.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while fetching the download URL.');
        });
});