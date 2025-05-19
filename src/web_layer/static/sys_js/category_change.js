// 按钮点击事件
document.querySelectorAll('#hotspotCard .category-btn').forEach(btn => {
  btn.addEventListener('click', function() {
      // 更新按钮状态
      document.querySelectorAll('#hotspotCard .category-btn').forEach(b => {
          b.classList.remove('active');
          b.classList.add('btn-outline-primary');
          b.classList.remove('btn-primary');
      });
      this.classList.add('active');
      this.classList.remove('btn-outline-primary');
      this.classList.add('btn-primary');
      
      // 加载对应数据
      var category = this.getAttribute('data-category');
      renderWordcloudCharts(output[category], categoryWordcloudChart, 'wordcloud-category' );
      renderPieCharts(output[category], categoryPieChart, 'pie-category');
      document.getElementById("pie_text").textContent = "热点分布图" + "(" + category + ")";
      document.getElementById("cloud_text").textContent = "热点词云图" + "(" + category + ")";
  });
});

// 按钮点击事件
document.querySelectorAll('#sentimentCard .category-btn').forEach(btn => {
  btn.addEventListener('click', function() {
      // 更新按钮状态
      document.querySelectorAll('#sentimentCard .category-btn').forEach(b => {
          b.classList.remove('active');
          b.classList.add('btn-outline-primary');
          b.classList.remove('btn-primary');
      });
      this.classList.add('active');
      this.classList.remove('btn-outline-primary');
      this.classList.add('btn-primary');
      
      // 加载对应数据
      var category = this.getAttribute('data-category');
    
      render_table(output[category], 'table-category');
      render_bar(output[category], 'bar-category');
      render_sentiment_text(category)
      //renderPieCharts(output[category], categoryPieChart, 'pie-category');
  });
});
