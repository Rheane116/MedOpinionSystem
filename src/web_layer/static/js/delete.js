function confirmDelete(button, itemName) {
    var userConfirmed = confirm("确认要删除" + itemName + "吗?");
    if (userConfirmed) {
                // 用户点击了“确定”，执行删除操作
                // 这里可以添加 AJAX 请求来通知服务器删除该项
                // 例如：
                // fetch('/delete-item', {
                //     method: 'POST',
                //     headers: {
                //         'Content-Type': 'application/json',
                //     },
                //     body: JSON.stringify({ item: itemName }),
                // })
                // .then(response => response.json())
                // .then(data => {
                //     if (data.success) {
                //         // 删除成功，从表格中移除该行
                //         button.parentElement.parentElement.remove();
                //     } else {
                //         alert("Delete failed!");
                //     }
                // })
                // .catch(error => {
                //     console.error('Error:', error);
                // });

                // 由于这是一个示例，我们直接移除表格行
                button.parentElement.parentElement.remove();
    }
}