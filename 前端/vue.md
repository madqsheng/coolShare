### QA

1. **Vue、JavaScript、HTML 和 CSS 是什么关系？**
   - HTML 是构建网页的基础语言，用于定义**网页的结构和内容**。它使用标签（例如 `<div>`、`<h1>`、`<p>` 等）来组织和展示文本、图像、视频等内容。
   - CSS 用于控制 HTML 元素的**样式和布局**。它通过选择器（如类、ID 或标签名）来应用样式规则，从而改变网页的外观。
   - JavaScript 是一种脚本语言，用于实现网页的动态行为和交互功能。它可以在用户与网页交互时更新内容、验证表单、创建动画等。
   - Vue.js 是一个渐进式 JavaScript 框架，用于构建用户界面。它通过使用组件和双向数据绑定来简化开发过程，使得开发复杂的单页应用（SPA）变得更加容易。
   - 总结：**Vue.js 本质上是用 JavaScript 编写的一个框架**。它的目的是让开发者能够更容易地将 JavaScript、HTML 和 CSS 结合起来，以便更高效地开发复杂的前端应用

2. **Vue的特点是什么？**

   - **渐进式框架**：

     - Vue.js 是一个渐进式框架，可以**从基础的用法逐步扩展到复杂的应用**。你可以只使用 Vue.js 来增强现有的项目，也可以使用它来构建整个单页应用（SPA）。

   - **组件化开发**：

     - Vue.js 允许开发者**将应用分解成可复用的组件**，**每个组件包含其 HTML 模板、CSS 样式和 JavaScript 逻辑**。这使得代码的组织更加清晰、维护更加方便。

   - **双向数据绑定**：

     - Vue.js 提供了双向数据绑定的功能，通过 `v-model` 指令可以轻松实现数据和视图的同步，这使得开发表单和交互式组件变得非常简单。

   - **指令系统**：

     - Vue.js 提供了许多内置指令（如 `v-if`、`v-for`、`v-bind`、`v-on` 等），这些指令可以直接在模板中使用，以实现条件渲染、列表渲染、事件处理等功能。

   - **响应式系统**：

     - Vue.js 具有强大的响应式系统，当数据发生变化时，视图会自动更新。Vue 的响应式系统基于观察者模式和依赖追踪，能够高效地处理数据和视图之间的同步。

   - **单文件组件（SFC）**：

     - Vue.js 支持单文件组件（`.vue` 文件），**在一个文件中定义组件的模板、脚本和样式。**这种方式使得组件的开发和维护更加方便。

     ```vue
     <template>
       <div class="my-component">
         <h1>{{ title }}</h1>
       </div>
     </template>
     
     <script>
     export default {
       data() {
         return {
           title: '这是一个标题'
         };
       }
     };
     </script>
     
     <style scoped>
     .my-component {
       color: #333;
     }
     </style>
     ```

   - **生态系统和工具链**：

     - Vue.js 拥有丰富的生态系统和工具链，包括 Vue Router（用于路由管理）、Vuex（用于状态管理）、Vue CLI（用于项目脚手架）等，使得开发、测试和部署 Vue 应用变得更加方便。

   - **声明式渲染**：

     - Vue.js 使用声明式语法来描述视图。当你用 Vue.js 编写模板时，你只需要专注于你想要在 UI 上呈现的最终状态，而不需要详细描述如何一步步实现这些状态的变化。Vue.js 会自动处理数据变化和视图更新之间的同步。

     ```vue
     <div id="app">
       <h1>{{ message }}</h1>
       <button @click="reverseMessage">反转消息</button>
     </div>
     
     <script>
     new Vue({
       el: '#app',
       data: {
         message: 'Hello, Vue!'
       },
       methods: {
         reverseMessage() {
           this.message = this.message.split('').reverse().join('');
         }
       }
     });
     </script>
     ```

### 1.基础

#### 1.1 HTML 中的常用属性

HTML 中有许多内置属性，以下是一些常用的属性：

- **通用属性**：
  - `id`: 元素的唯一标识符。
  - `class`: 元素的样式类名，可以是多个类名。
  - `style`: 内联样式。
  - `title`: 鼠标悬停时显示的文本。
  - `lang`: 元素内容的语言代码。
  - `data-*`: 用于存储页面或应用程序的私有定制数据。
- **表单元素属性**：
  - `type`: 定义 `<input>` 元素的类型（如 `text`, `password`, `checkbox`）。
  - `name`: 表单字段的名称。
  - `value`: 表单字段的初始值。
  - `placeholder`: 提示用户输入内容的简短描述。
  - `checked`: 适用于 `<input type="checkbox">` 和 `<input type="radio">`，表示是否选中。
  - `disabled`: 禁用元素，使其不可编辑。
  - `readonly`: 使输入字段为只读。
- **锚点和链接属性**：
  - `href`: 超链接目标的 URL。
  - `target`: 指定在何处打开链接文档（如 `_blank`, `_self`）。
- **图像属性**：
  - `src`: 图像文件的 URL。
  - `alt`: 图像不可用时显示的替代文本。
  - `width`: 图像的宽度。
  - `height`: 图像的高度。
- **表格属性**：
  - `colspan`: 单元格跨越的列数。
  - `rowspan`: 单元格跨越的行数。
- **多媒体属性**：
  - `controls`: 是否显示播放控件（适用于 `<audio>` 和 `<video>`）。
  - `autoplay`: 是否自动播放（适用于 `<audio>` 和 `<video>`）。
  - `loop`: 是否循环播放（适用于 `<audio>` 和 `<video>`）。
  - `muted`: 是否静音（适用于 `<audio>` 和 `<video>`）。

#### 1.2 CSS如何选择HTML元素

1. **标签选择器**

   选择所有指定标签的元素。

   ```html
   <p>这是一个段落。</p>
   <p>这是另一个段落。</p>
   ```

   ```css
   /* 选择所有 <p> 元素 */
   p {
     color: blue;
   }
   ```

2. **类选择器**

   选择所有具有指定类名的元素。类选择器使用 `.` 表示。

   ```html
   <div class="container">
       这是一个容器。
    </div>
    <div class="container">
       这是另一个容器。
    </div>
   ```

   ```css
   /* 选择所有具有类名 "container" 的元素 */
   .container {
     background-color: #f0f0f0;
   }
   ```

3. **ID 选择器**

   选择具有指定 ID 的元素。ID 选择器使用 `#` 表示。

   ```html
   <div id="header">
       这是标题。
   </div>
   ```

   ```css
   /* 选择 ID 为 "header" 的元素 */
   #header {
     font-size: 24px;
   }
   ```

4. **属性选择器**

   选择具有指定属性的元素，可以指定属性值。属性选择器使用方括号 `[]` 表示。

   ```html
   <p title="example">这个段落有一个 title 属性。</p>
   <p title="another">这个段落有另一个 title 属性。</p>
   ```

   ```css
   /* 选择所有具有 title 属性的元素 */
   [title] {
     color: red;
   }
   
   /* 选择所有 title 属性值为 "example" 的元素 */
   [title="example"] {
     color: green;
   }
   ```

5. **伪类选择器**

   选择**元素的某个特定状态**，如悬停、选中等。常用的伪类有 `:hover`、`:active`、`:focus` 等。

   ```html
   <a href="#">这是一个链接</a>
   <br>
   <input type="text" placeholder="输入框">
   ```

   ```css
   /* 选择所有悬停状态的 <a> 元素 */
   a:hover {
     text-decoration: underline;
   }
   
   /* 选择所有焦点状态的输入框 */
   input:focus {
     border-color: blue;
   }
   ```

6. **伪元素选择器**

   选择元素的**某个部分**，如第一个字母、第一行等。常用的伪元素有 `::before`、`::after`、`::first-letter`、`::first-line` 等。

   ```html
   <p>这是一个段落。</p>
   <p>这是另一个段落。</p>
   ```

   ```css
   /* 在每个 <p> 元素的内容前添加内容 */
   p::before {
     content: "Note: ";
     color: red;
   }
   
   /* 选择所有 <p> 元素的第一个字母 */
   p::first-letter {
     font-size: 200%;
     color: blue;
   }
   ```

7. **组合选择器**

   组合选择器用于**选择特定关系的元素**，包括后代选择器、子选择器、相邻兄弟选择器和通用兄弟选择器。

   - **后代选择器**（空格）：选择某元素的**所有后代元素。**

     ```css
     /* 选择所有 <div> 元素内的 <p> 元素 */
     div p {
       color: purple;
     }
     ```

   - **子选择器**（`>`）：选择某元素的**直接子元素。**

     ```css
     /* 选择所有 <ul> 元素的直接子 <li> 元素 */
     ul > li {
       list-style-type: none;
     }
     ```

   - **相邻兄弟选择器**（`+`）：选择**某元素的下一个兄弟元素**。

     ```css
     /* 选择所有 <h1> 元素后面的第一个 <p> 元素 */
     h1 + p {
       margin-top: 0;
     }
     ```

   - **通用兄弟选择器**（`~`）：选择某元素之后的所有兄弟元素。

     ```css
     /* 选择所有 <h1> 元素后面的所有 <p> 元素 */
     h1 ~ p {
       color: gray;
     }
     ```

8. **组合选择器**

   组合多个选择器以更精确地选择元素。

   ```css
   /* 选择所有类名为 "button" 且悬停状态的元素 */
   .button:hover {
     background-color: lightblue;
   }
   
   /* 选择所有 ID 为 "header" 且类名为 "main-header" 的元素 */
   #header.main-header {
     font-weight: bold;
   }
   ```

**示例应用**

假设有以下 HTML 结构：

```
html复制代码<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CSS Selectors Example</title>
  <style>
    /* 使用上述选择器样式 */
    p {
      color: blue;
    }
    
    .container {
      background-color: #f0f0f0;
    }

    #header {
      font-size: 24px;
    }

    [title] {
      color: red;
    }

    a:hover {
      text-decoration: underline;
    }

    p::before {
      content: "Note: ";
      color: red;
    }

    div p {
      color: purple;
    }

    ul > li {
      list-style-type: none;
    }

    h1 + p {
      margin-top: 0;
    }

    h1 ~ p {
      color: gray;
    }

    .button:hover {
      background-color: lightblue;
    }

    #header.main-header {
      font-weight: bold;
    }
  </style>
</head>
<body>
  <div id="header" class="main-header">Header</div>
  <div class="container">
    <p title="example">This is a paragraph.</p>
    <p>This is another paragraph.</p>
    <a href="#">This is a link</a>
    <ul>
      <li>List item 1</li>
      <li>List item 2</li>
    </ul>
    <h1>Heading</h1>
    <p>Paragraph after heading</p>
  </div>
</body>
</html>
```

#### 1.3 Vue如何选择HTML元素

1. **使用 `el` 属性选择根元素**

   在 Vue 实例中，`el` 属性用于绑定 Vue 实例到一个 DOM 元素。可以通过 `id` 或者 `class` 选择器来选择元素。

   ```html
   <!DOCTYPE html>
   <html lang="en">
   <head>
     <meta charset="UTF-8">
     <meta name="viewport" content="width=device-width, initial-scale=1.0">
     <title>Vue 使用 el 属性选择根元素</title>
   </head>
   <body>
     <div id="app">{{ message }}</div>
     <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
     <script>
       new Vue({
         el: '#app',
         data: {
           message: 'Hello, Vue!'
         }
       });
     </script>
   </body>
   </html>
   
   ```

   在这个例子中，Vue 实例通过 `el: '#app'` 选择了 `id` 为 `app` 的 `div` 元素，并绑定了 `message` 数据属性。

2. **使用模板语法绑定数据**

   Vue 允许在模板中使用 Mustache 语法 `{{ }}` 来绑定数据。

   ```html
   <!DOCTYPE html>
   <html lang="en">
   <head>
     <meta charset="UTF-8">
     <meta name="viewport" content="width=device-width, initial-scale=1.0">
     <title>Vue 使用模板语法绑定数据</title>
   </head>
   <body>
     <div id="app">
       <p>{{ message }}</p>
     </div>
     <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
     <script>
       new Vue({
         el: '#app',
         data: {
           message: 'Hello, Vue with Template!'
         }
       });
     </script>
   </body>
   </html>
   
   ```

3. **使用 `ref` 属性引用元素**

   可以使用 `ref` 属性来引用 DOM 元素，然后通过 `this.$refs` 访问这些元素。

   ```html
   <!DOCTYPE html>
   <html lang="en">
   <head>
     <meta charset="UTF-8">
     <meta name="viewport" content="width=device-width, initial-scale=1.0">
     <title>Vue 使用 ref 属性引用元素</title>
   </head>
   <body>
     <div id="app">
       <p ref="paragraph">{{ message }}</p>
       <button @click="changeMessage">改变消息</button>
     </div>
     <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
     <script>
       new Vue({
         el: '#app',
         data: {
           message: 'Hello, Vue with Ref!'
         },
         methods: {
           changeMessage() {
             this.message = 'Message Changed!';
             this.$refs.paragraph.style.color = 'red'; // 改变引用元素的样式
           }
         }
       });
     </script>
   </body>
   </html>
   
   ```

   

- 官网：[Vue.js中文网 (zcopy.site)](https://vue2.zcopy.site/)

### 2.开发范式

- 开发环境

- 脚手架使用
- vscode插件
- 浏览器扩展

### 3.常用的库

### 4.包管理