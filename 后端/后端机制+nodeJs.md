# 1.  概述

- node.js用作服务端，比如：小程序，手机APP，网站，游戏
- 通俗的说，node.js要做的事是：让每一个人访问我做的应用，比如网站
- node.js是什么？
  - node.js并不是编程语言
  - node.js是一款应用程序，或者说软件
  - node.js可以运行JavaScript代码，但是它运行的js代码和开发网页的js有些不一样
- 题外话：
  - 软件开发中的两种架构
    1. C/S——客户端/服务器架构
    2. B/S——浏览器/服务器架构
  - node.js代码运行在服务器，属于B/S架构。
  - 服务器先保存HTML，css，JavaScript代码，浏览器通过URL向服务器发送请求，运行在服务器的node.js代码会将服务器的资源返回给服务器。

# 2. HTTP协议

## 2.1 概述

- Hyper Text Transfer Protocol 超文本传输协议，本质是一系列规则。
- 将HTML（Hypertext Markup Language）文档从web服务器传送到客户端浏览器
- 这个协议详细规定了 浏览器和==万维网==（服务器 ）之间互相通信的规则
- 属于应用层的面向对象的协议

## 2.2 与TCP/IP关系

![TCP_IP](..\实例图片\TCP_IP.jpg)

- Transmission Control Protocol/Internet Protocol 传输控制协议/网际协议
- TCP/IP协议其实是一系列与互联网相关联的协议集合总称
- TCP/IP协议不仅仅指的是TCP 和IP两个协议，而是指一个由FTP、[SMTP](https://so.csdn.net/so/search?q=SMTP&spm=1001.2101.3001.7020)、TCP、UDP、IP等协议构成的协议簇， 只是因为在TCP/IP协议中TCP协议和IP协议最具代表性，所以被称为TCP/IP协议。
- 关系：HTTP协议是构建在TCP/IP协议之上的，是TCP/IP协议的一个子集

## 2.3 HTTP协议的内容

- http协议本质是通信规则，包括请求结构，请求方法，各种机制等。这些全部体现在http报文中

- 协议中主要规定了两个方面的内容

  - 客户端：用来向服务器发送数据，可以被称之为请求报文
  - 服务端：向客户端返回数据，可以被称之为响应报文

- http报文内容

  - 请求报文

    - 请求行：请求方法+URL+http版本号
    - 请求头：主要记录浏览器的相关信息和交互行为（协议机制）
      - 参考：[HTTP 标头（header） - HTTP | MDN (mozilla.org)](https://developer.mozilla.org/zh-CN/docs/web/http/headers)

    - 空行

    - 请求体：向服务器发送请求的内容
  - 响应报文

    - 响应行：http版本号+响应状态码+响应状态描述
      - 参考：[HTTP 响应状态码 - HTTP | MDN (mozilla.org)](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Status)
    - 响应头：主要记录服务器和响应体的相关信息
      - 参考：[HTTP 标头（header） - HTTP | MDN (mozilla.org)](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Headers)
    - 空行
    - 响应体：服务器提供的资源，内容的类型是非常灵活的，常见的类型有 HTML、CSS、JS、图片、JSON

- node.js中使用http协议的是内置的http模块

  ```js
  // 1. 导入http模块
  const http = require('http')
  
  // 2. 创建服务对象
  const server = http.createServer((request,response)=>{
      response.setHeader('content-type','text/html;charset=utf-8')
      response.end('Hello HTTP Server，你好')
  
  })
  
  // 3.监听端口，启动服务
  server.listen(9000,()=>{
      console.log('服务启动了')
  })
  ```

## 2.4 请求报文

### 2.4.1 HTTP请求行

- 请求方法（get、post、put、delete等）

- 请求 URL（统一资源定位器）

  例如：https://www.baidu.com/index.html?a=100&b=200#logo

  - http： 协议（https、ftp、ssh等）
  - www.baidu.com ：域名
  - 80：端口号
  - /index.html ：路径（==路径里可能有params参数==）
  - a=100&b=200 ：query查询字符串
  - \#logo ：哈希（锚点链接）

- HTTP协议版本号

- ==浏览器除了请求体给服务器传递数据，还可以通过请求行中URL路径中的的query和params参数==，[详细跳转](##2.8 query和params参数)

### 2.4.2 HTTP请求头

- 格式：『头名：头值』
- 参考：[HTTP 标头（header） - HTTP | MDN (mozilla.org)](https://developer.mozilla.org/zh-CN/docs/web/http/headers)

### 2.4.3 HTTP请求体

- 请求体内容的格式是非常灵活的
  - 可以是空，比如：GET请求
  - 也可以是字符串，还可以是JSON，比如：POST请求

## 2.5 响应报文

- 响应行：比如：`HTTP/1.1 200 OK`
  - HTTP/1.1：HTTP协议版本号
  - 200：响应状态码，还有一些状态码，[参考](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Status)
  - OK：响应状态描述
- 响应头：主要记录服务器和响应体的相关信息
- 空行
- 响应体：响应体内容的类型是非常灵活的，常见的类型有 HTML、CSS、JS、图片、JSON

## 2.6 node.js中实践http协议的思路

- 概述

  - node.js程序是运行在服务器的
  - HTML，CSS，JavaScript是放在服务器的资源
  - 浏览器会向服务器发送请求，比如：FORM表单标签，ajax
  - node.js程序就是用于处理浏览器向服务器发送的请求。决定服务器给浏览器返回具体什么资源。

- node.js决定给浏览器返回资源，主要依靠的是请求头里的路径和请求方法：

  ```javascript
  //导入 http 模块
  const http = require('http');
  const fs = require('fs');
  
  //创建服务对象
  const server = http.createServer((request, response) => {
    //获取请求url的路径
    let {pathname} = new URL(request.url, 'http://127.0.0.1');
    if(pathname === '/'){
      //读取文件内容
      let html = fs.readFileSync(__dirname + '/10_table.html');
      response.end(html); //设置响应体
    }else if(pathname === '/index.css'){
      //读取文件内容
      let css = fs.readFileSync(__dirname + '/index.css');
      response.end(css); //设置响应体
    }else if(pathname === '/index.js'){
      //读取文件内容
      let js = fs.readFileSync(__dirname + '/index.js');
      response.end(js); //设置响应体
    }else{
      response.statusCode = 404;
      response.end('<h1>404 Not Found</h1>')
    }
    
  });
  
  //监听端口, 启动服务
  server.listen(9000, () => {
    console.log('服务已经启动....')
  });
  
  ```

  - 具体过程：
    1. 浏览器地址输入：127.0.0.1:9000，第一次请求，服务器返回10_table.html
    2. 而在10_table.html里面引用了index.css文件，这会自动触发第二次请求，服务器返回index.css
    3. 继续10_table.html里面引用了index.js文件，这会自动触发第二次请求，服务器返回index.js

- [关联知识](###4.3.1 express-generator工具生成结构)

## 2.7 GET和POST请求

- get请求场景
  - 浏览器地址栏输入url
  - 点击a链接，img标签引入图片
  - link标签引入css，script标签引入js
  - video和audio引入媒体
  - form标签中的method为get（不区分大小）
  - ajax中的get请求
- post请求场景
  - form标签中的method为post（不区分大小）
  - ajax中的post请求

- get和post请求的区别
  - 两者都是http协议内容里的请求方式
  - 主流来说，get主要作用是获取数据，post主要是提交数据。
  - 一般来说，get带参数请求是将参数放到URL后，post带参数请求是将参数放到请求体
  - 相对来说，post请求会比get请求更安全，因为浏览器中的URL参数会暴露在地址栏
  - get请求有大小限制，一般为2k，而post请求则没有大小限制

## 2.8 query和params参数

- 概述

  1. 浏览器想给服务器传数据，如果比较轻量，虽然用请求体传数据有点浪费，勉强还行。那么如果数据要体现在路由里的路径呢？请求体无能为力。

  2. 实现传递数据体现在路由里的路径，方法是在路径里加入query和params参数实现传数据：

     浏览器输入：http://127.0.0.1/creat?a=100&b=200

     

- node中应用

  - query参数

    1. 设置路由规则，获取参数

       ```js
       // 删除记录
       router.get('/account',(req,res) =>{
       	let id = req.query.id
           //mongodb
           AccountsModel.deleteOne({
             _id:id
           }).catch((err) =>{
             console.log(err)
             throw err
           }).then((data) =>{
             console.log('删除成功!!')
             res.render('success',{msg:'删除成功~',url:'/account'})
           })
       })
       ```

       

    2. 浏览器输入：http://127.0.0.1/account?id=100&count=200，或者其他get请求的途径

    3. query参数有明显的特征：?和&，因此query参数不太安全

    4. 前面服务端1中路由规则里的`req.query.id`其实就是100，`req.query.id`是200

  - params参数

    1. 设置路由规则，获取参数

       ```js
       // 删除记录
       router.get('/account/:id',(req,res) =>{
         // 获取params的id参数
         let id = req.params.id
       
         //lowdb
         // // 删除对应数据
         // db.get('accounts').remove({id:id}).write()
       
         //mongodb
         AccountsModel.deleteOne({
           _id:id
         }).catch((err) =>{
           console.log(err)
           throw err
         }).then((data) =>{
           console.log('删除成功!')
           res.render('success',{msg:'删除成功~',url:'/account'})
         })
       })
       ```

       

    2. 浏览器输入：http://127.0.0.1/account/123 或者其他get请求的途径

       - 其中的123不是路径，而是参数
       - 看到一个这样的url，辨别123是否是参数，其实很难，只能去看服务端的路由代码。正因如此，params参数相对比query参数安全

    3. 前面服务端1中路由规则里的`req.params.id`其实就是123

==注意，query和params参数并不受请求方法的限制。无论使用GET方法还是POST方法，都可以获取和传递这些参数==

## 2.9 局域网ip、广域网ip、localhost

- ip本身是一个数字标识，用来标志网络设备，实现设备间通信

- ip本身是一个32bit数据，分为四组，每组0-255

- 局域网背景：

  1. 32bit数据可以有42（4294967296）亿多ip，有点不太够
  2. 某个区域内共享ip的机制—局域网ip或者私网IP

- 广域网ip或者公网ip

- 本地回环ip

  - 指向当前本机
  - 不只常见127.0.0.1

  | 类型               | 说明                                                         |
  | ------------------ | ------------------------------------------------------------ |
  | 本地回环IP         | 127.0.0.1 ~ 127.255.255.254                                  |
  | 局域网IP（私网IP） | 192.168.0.0 ~ 192.168.255.255<br>172.16.0.0 ~ 172.31.255.255<br/>10.0.0.0 ~ 10.255.255.255 |
  | 广域网IP（公网IP） | 除上述之外                                                   |

  

# 3 开发生态

## 3.1 模块化

- 主架流程(本质是导入js文件)

  1. 模块文件：声明一个函数作为目标，module.exports暴露

     me.js

     ```javascript
     //声明一个函数
     function tiemo(){
       console.log('贴膜...');
     }
     
     //捏脚
     function niejiao(){
       console.log('捏脚....');
     }
     
     //暴露数据
     module.exports = {
       tiemo,
       niejiao
     }
     ```

  2. 应用文件：require导入模块，调用目标函数

     index.js

     ```javascript
     //导入模块
     const me = require('./me.js');
     
     //输出 me
     console.log(me);
     file.tiemo();
     me.niejiao();
     ```

- 导入文件夹

  - 本质还是导入文件夹下面的文件，不过有严格的预设的规则

    1. 应用文件：require导入文件夹

       main.js

       ```javascript
       //导入
       const m = require('./module');
       
       console.log(m);
       ```

    2. 被引入的module文件夹的结构预设规则：

       - 首先检测文件夹下的package.json文件，读取里面的main属性，在找到对应的文件

         ```json
         {
         	"main":"./me.js"
         }
         ```

       - 如果package.json文件不存在，则会直接导入文件夹下的**index.js**和**index.json**，如何还没有找到，则会报错。

## 3.2包管理

- 概述
  1. 包：package，代表了一组特定功能的源码集合
  2. 包管理工具：应用软件，对包进行下载安装，更新，删除，上传等操作
  3. 包括：npm（官方内置），yarn，cnpm
- [关联](#5. npm "具体介绍")

## 3.3 NVM：node版本管理

- nvm 全称 `Node Version Manager` 顾名思义它是用来管理 node 版本的工具，方便切换不同版本的 Node.js

- 使用：

  1. 下载安装

     - 下载地址： https://github.com/coreybutler/nvm-windows/releases，选择 nvm-setup.exe 下载即可

  2. 常用命令

     | 命令                  | 说明                            |
     | :-------------------- | :------------------------------ |
     | nvm list available    | 显示所有可以下载的 Node.js 版本 |
     | nvm list              | 显示已安装的版本                |
     | nvm install 18.12.1   | 安装 18.12.1 版本的 Node.js     |
     | nvm install latest    | 安装最新版的 Node.js            |
     | nvm uninstall 18.12.1 | 删除某个版本的 Node.js          |
     | nvm use 18.12.1       | 切换 18.12.1 的 Node.js         |

     

# 4 技术生态

## [4.1 HTTP协议](#2 HTTP协议)

## [4.2 包管理工具](##3.2包管理)

## 4.3 Express框架

- [Express 中文网 (nodejs.cn)](https://express.nodejs.cn/)

- [关联知识](##2.6 node.js中实践http协议的思路)
  - 原生的node.js实践http协议，三步：
    1. 设置请求方法和路径
    2. 通过路径和方法找到资源
    3. 读取资源返还给浏览器
  - 甚至对于动态资源的处理也很麻烦

### 4.3.1 express-generator工具生成结构

- 安装：`npm install -g express-generator`

  全局安装以后，会暴露一个express命令

- 创建项目主架：

  `Usage: express [options] [dir]`

  `express -e accounts`

- 安装依赖

  ```cmd
  cd accounts
  npm install
  ```

  生成五个文件夹和一系列文件：

  1. bin文件夹下www.js是入口文件
  2. node_modules文件夹下是包源代码
  3. public文件夹下是静态资源
  4. routes文件夹下是路由规则配置
  5. views文件夹下是模板引擎文件
  6. app.js文件是导入路由，静态文件，模板引擎等，暴露给www.js调用
  7. package.json和package-lock.json文件是包管理文件

- express架构分为：路由，中间件，模板引擎

  express代码主干是：创建对象，路由规则，监听端口

  ```cmd
  //1. 导入 express
  const express = require('express');
  
  //2. 创建应用对象
  const app = express();
  
  //3. 创建路由规则
  app.get('/home', (req, res) => {
  res.end('hello express server');
  });
  
  //4. 监听端口 启动服务
  app.listen(3000, () =>{
  console.log('服务已经启动, 端口监听为 3000...');
  });
  ```

  - 中间件是一个贯穿始终、极为重要的概念。本质是一个回调函数。

    ==express框架里，几乎什么都可以作为中间件，甚至写好的路由规则也会被封装成某个路由的中间件==，比如：route/index.js下里面很多路由规则

    ```js
    var express = require('express');
    var router = express.Router();
    
    // 记账本的列表
    router.get('/account', function(req, res, next) {
      // 获取所有的账单信息
      let accounts = db.get('accounts').value()
      // console.log(accounts)
      res.render('list',{accounts:accounts});
    });
    
    // 添加记录
    router.get('/account/create', function(req, res, next) {
      res.render('create');
    });
    
    
    module.exports = router;
    
    ```

    会成为app.js里的中间件：

    ```js
    // 导入index.js文件
    var indexRouter = require('./routes/index');
    
    // indexRouter成了中间件
    app.use('/', indexRouter);
    ```

  - 所以，==express架构分为：路由，中间件，模板引擎==，其他很多东西都包含在中间件中，比如静态资源等

### 4.2.2 路由规则管理

- 路由是http里的概念，一个路径和方法对应一种资源，也就是对应浏览器的请求req和服务器的响应res

- 很多时候，路由和中间件是结合在一起

  ```js
  const express = require('express')
  const app = express()
  
  // 路径--资源
  app.get('/', (req, res) => {
    res.send('hello world')
  })
  
  //路径--中间件,任何请求类型
  //这个例子展示了一个挂载在 /user/:id 路径上的中间件函数。 该函数针对 /user/:id 路径上的任何类型的 HTTP 请求执行。
  app.use('/user/:id', (req, res, next) =>{
    console.log('Request Type:',req.method)
    next()
  })
  
  //路径--中间件，限定请求类型
  //这个例子展示了一个路由和它的处理函数（中间件系统）。 该函数处理对 /user/:id 路径的 GET 请求。
  app.get('/user/:id', (req, res, next) =>{
    res.send('USER')
  })
  ```

- 路由管理模块化方法：

  - 具体业务路由，放在[express-generator工具](###4.3.1 express-generator工具生成结构)下的routes文件夹下

    1. 导入express，引入Router
    2. 设置路由规则，中间件格式：

    ```js
    var express = require('express');
    var router = express.Router();
    
    
    // 添加记录
    router.get('/account/create', function(req, res, next) {
      res.render('create');
    });
    
    module.exports = router;
    ```

  - 然后在引用文件app.js文件中，将routes目录下文件管理的路由当做中间件

    ```js
    // 导入index.js文件
    var indexRouter = require('./routes/index');
    
    // indexRouter成了中间件
    //路径是/，方式不限，得到的资源或者中间件是indexRouter
    app.use('/', indexRouter);
    ```

### 4.3.3 中间件

- 概述：

  1. 明确一点，中间件和路由自始至终都密不可分
  2. 中间件本质是回调函数，作用是使用函数封装公共操作，简化代码
  3. 中间件函数是应用在http协议的请求-响应周期中。中间件函数可以访问 [请求对象](https://express.nodejs.cn/en/4x/api.html#req) (`req`)、[响应对象](https://express.nodejs.cn/en/4x/api.html#res) (`res`) 和 `next` 函数。
  4. 在http协议的请求-响应周期中，如果没有结束请求-响应循环，它必须调用 `next()` 将控制权传递给下一个中间件函数。 否则，请求将被挂起

- 从作用域来分，全局中间件和路由私有中间件

  1. 中间件本质是回调函数，所以先定义一个参数为：`req, res, next`的函数`myFunction`
  2. 全局作用中间件：`app.use(myFunction)`
  3. 特定路由私有中间件：`app.get('/',myFunction,(req, res) => {})`

  ```js
  const express = require('express')
  const app = express()
  
  //函数，参数：req, res, next
  const myLogger = function (req, res, next) {
    console.log('LOGGED')
    next()
  }
  
  const myFunction = function (req, res, next) {
    console.log('myFunction')
    next()
  }
  
  //全局中间件
  app.use(myLogger)
  
  //myFunction是特定路由私有中间件
  app.get('/', myFunction，(req, res) => {
    res.send('Hello World!')
  })
  
  //解释：路由app.get('/', myFunction，(req, res) => {})
  //回调函数是(req, res) => {}，会在此之前执行myLogger和myFunction中间件函数
  
  ```

  

- 从功能来说，有很多中间件，都是为了辅助http协议的进行，或者说辅助路由过程

  1. 静态文件路由

     - 中间件函数：`express.static(root, [options])`,`root` 参数指定提供静态资源的根目录

     - 应用：`app.use(express.static('public'))`

       app.js:

       ```js
       var express = require('express');
       var path = require('path');
       var app = express();
       
       //注意这里__dirname是获取当前文件的目录绝对位置，这样运行代码跟启动 node 进程的目录无关了。
       //原因是，app.use(express.static('public'))获取的是启动node服务的位置下的public目录，但是我们无法保证就在app.js同级目录下启动node。而__dirname是获得app.js所在目录的绝对路径
       app.use(express.static(path.join(__dirname, 'public')))
       ```

     - 路由寻找静态资源只需要给出相对path.join(__dirname, 'public')的路径就可以

       ```html
       <!DOCTYPE html>
       <html lang="en"> 
         <head>
             ...
             ...
           <link
             href="/css/bootstrap.css"
             rel="stylesheet"
           />
           <lnk href="/css/bootstrap-datepicker.css" rel="stylesheet">
         </head>
         <body>
           ...
           ...
           </div>
           <script src="/js/jquery.min.js"></script>
           <script src="/js/bootstrap.min.js"></script>
           <script src="/js/bootstrap-datepicker.min.js"></script>
           <script src="/js/bootstrap-datepicker.zh-CN.min.js"></script>
           <script src="/js/main.js"></script>
         </body>
       </html>
       ```

       - public目录下放`js/jquery.min.js`，`/css/bootstrap.css`等放对应的资源就行 

  2. 处理的cookie中间件函数

  3. 

- 从绑定对象来分，有`app = express()`和`express.Router()`两个对象

  - 工作方式 `router.use()` 和 `router.METHOD()` --- `app.use()` 和 `app.METHOD()`

  - 作用域不一样

### 4.3.4 模板引擎

- 概述

  - 模板引擎是分离用户界面和业务数据 的一种技术
  - 简单说，就是html和服务端js（变量）分开
  - 在运行时，模板引擎将模板文件中的变量替换为实际值，并将模板转换为发送给客户端的 HTML 文件
  - 特定语法

- 使用方法：

  1. 安装：`npm i ejs`

  2. 设置模板引擎类型和模板文件存放位置

     app.js

     ```js
     var path = require('path');
     
     // 设置模板文件存放位置
     app.set('views', path.join(__dirname, 'views'));
     // 设置模板引擎类型
     app.set('view engine', 'ejs');
     ```

  3. 当前目录下的`views`文件夹里是模板文件

     success.ejs

     ```ejs
     <!DOCTYPE html>
     <html lang="en">
     <head>
       <meta charset="UTF-8">
       <meta http-equiv="X-UA-Compatible" content="IE=edge">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title>提醒</title>
       <link
           href="https://cdn.bootcdn.net/ajax/libs/twitter-bootstrap/3.3.7/css/bootstrap.css"
           rel="stylesheet"
         />
       <style>
         .h-50{
           height: 50px;
         }
       </style>
     </head>
     <body>
       <div class="container">
         <div class="h-50"></div>
         <div class="alert alert-success" role="alert">
           <h1>:) <%= msg%></h1>
           <p><a href="<%= url%>">点击跳转</a></p>
         </div>
       </div>
     </body>
     </html>
     ```

  4. 路由管理下渲染模板，传入数据 `res.render('success',{msg:'删除成功~',url:'/account'})`

     routes/index.js

     ```js
     var express = require('express');
     var router = express.Router();
     
     // 导入lowdb
     const low = require('lowdb')
     const FileSync = require('lowdb/adapters/FileSync')
      
     // 获取db对象
     const adapter = new FileSync(__dirname+'/../data/db.json')
     const db = low(adapter)
     
     const shortid = require('shortid')
     
     
     
     // 删除记录
     router.get('/account/:id',(req,res) =>{
       // 获取params的id参数
       let id = req.params.id
     
       // 删除对应数据
       db.get('accounts').remove({id:id}).write()
     
       res.render('success',{msg:'删除成功~',url:'/account'})
     
     })
     
     module.exports = router;
     
     ```

  5. [ejs语法](https://ejs.bootcss.com/#install)

### 4.3.5 Lowdb数据库

- 概述

  1. lowdb是一个本地的json文件数据库，简单来说就是用一个json文件来充当数据库，来实现增删改查这些数据库的基本的功能
  2. 安装npm i lowdb@1.0.0

- 配合表单请求和数据处理

  1. 模板文件中，表单标签添加name属性，方便服务器处理表单数据，其中select标签的option的value设置值
  2. form标签设置请求方式和url
  3. 给form的请求方式和url配置路由

  ```ejs
  ...
  <form method="post" action="/account">
      <div class="form-group">
          <label for="item">事项</label>
          <input
                 name="title"
                 type="text"
                 class="form-control"
                 id="item"
                 />
      </div>
      <div class="form-group">
          <label for="time">发生时间</label>
          <input
                 name="time"
                 type="text"
                 class="form-control"
                 id="time"
                 />
      </div>
      <div class="form-group">
          <label for="type">类型</label>
          <select name="type" class="form-control" id="type">
              <option value="-1">支出</option>
              <option value="1">收入</option>
          </select>
      </div>
      <div class="form-group">
          <label for="account">金额</label>
          <input
                 name="account"
                 type="text"
                 class="form-control"
                 id="account"
                 />
      </div>
  
      <div class="form-group">
          <label for="remarks">备注</label>
          <textarea name="remarks" class="form-control" id="remarks"></textarea>
      </div>
      <hr>
      <button type="submit" class="btn btn-primary btn-block">添加</button>
  </form>
  ...
  ```

  4. 导入lowdb，获取db对象，初始化db，新建一个数据结构(比如：数组)保存数据
  5. 保存数据
  6. 借助shortid工具包生成独一无二的id `npm i shortid`

  ```js
  // 导入lowdb
  const low = require('lowdb')
  const FileSync = require('lowdb/adapters/FileSync')
   
  // 获取db对象
  const adapter = new FileSync(__dirname+'/../data/db.json')
  const db = low(adapter)
  //导入shortid
  const shortid = require('shortid')
  
  router.post('/account',(req,res) =>{
    // 获取请求
    console.log(req.body)
  
    // 初始化db，新建一个accounts的数组
    db.defaults({ accounts: []}).write()
  
    let id = shortid.generate()
  
    // 写入文件
    db.get('accounts').unshift({id:id,...req.body}).write()
  
    //写入记录
    res.send('添加记录')
  })
  ```

## 4.4 数据库MongoDB

- 概述

  1. 基于分布式文件存储的数据库
  2. 管理数据，增删改查
  3. 相比于纯文件管理数据，数据库管理数据有如下特点：
     - 速度更快
     - 扩展性更强
     - 安全性更强

- 核心概念：

  - 数据库（database） 数据库是一个数据仓库，数据库服务下可以创建很多数据库，数据库中可以存 放很多集合
  - 集合（collection） 集合类似于 JS 中的数组，在集合中可以存放很多文档 
  - 文档（document） 文档是数据库中的最小单位，类似于 JS 中的对象

  ![mongodb架构](..\实例图片\mongodb架构.png)

### 4.4.1 下载安装

- [下载地址](https://www.mongodb.com/try/download/community-kubernetes-operator)

  - 压缩包自定义安装(.zip)

    1. 解压

    2. 数据库保存位置

       - 自定义数据库保存位置

         1. 自定义创建文件夹`data/db`

         2. 启动服务器的时候设置自定义位置，每次都要这样

            解压mongodb里的bin文件夹下：`mongod --dbpath ../data/db`

       - 数据库默认保存位置（方便推荐）

         1. windows系统下，数据库默认保存位置取决于启动mongodb命令的位置

         2. 比如启动命令位置：`E:\mongodb\mongodb-win32-x86_64-windows-6.0.7\bin`

            数据库默认保存位置:`E:\data\db`

         3. 需要自己常见data和db文件夹，否则运行`mongodb`会报错，提示找不到`E:\data\db`

         4. 解压mongodb里的bin文件夹下：`mongod`

    3. 还可以配置mongodb数据库服务，目的是数据库随电脑自启

       [参考](https://blog.csdn.net/weixin_43898497/article/details/115452745)

  - 安装包安装(.mis)

    - 一步步点击安装就行，会自动创建数据库保存位置和配置mongodb数据库服务
    - 适合服务器部署阶段

- 图形化管理工具

  - [Robo 3T 免费]( https://github.com/Studio3T/robomongo/releases)
  - [Navicat 收费](https://www.navicat.com.cn/) 

### 4.4.2 mongoose代码操作mongodb

[参考](https://blog.csdn.net/weixin_45828332/article/details/114120710)

- 概述

  1. Mongoose 是一个可以通过Node来操作MongoDB数据库的一个模块
  2. Mongoose 是一个对象文档模型（ODM）库，它是对Node原生的MongoDB模块进行了进一步的优化封装

- 基本流程

  1. 安装mongoose：`npm i mongoose`

  2. 导入mongoose：`const mongoose = require('mongoose')`

  3. 连接mongoose服务，包括mongodb下协议，ip，端口，数据库名：

     `mongoose.connect('mongodb://127.0.0.1:27017/bilibili')`

  4. 给服务不同结果设置回调，connection后的事件绑定继承[EventEmitter]([Events | Node.js v20.4.0 Documentation (nodejs.org)](https://nodejs.org/api/events.html#class-eventemitter))，函数里第一个参数是[事件关键词](https://mongoosejs.com/docs/connections.html)对应不同的结果。第二个参数是回调函数，里面开始操作数据库里的增删改查：

     - 成功：`mongoose.connection.on('open', () => {})`
     - 出错：`mongoose.connection.on('error', () => {})`
     - 关闭：`mongoose.connection.on('close', () => {})`

  5. 回调函数里写业务逻辑里的增删改查

     - 创建文档结构Schema对象，设置文档字段和数据类型

       ```js
       let BookSchema = new mongoose.Schema({
           title: String,
           author: String,
           price: Number
       });
       ```

     - 创建文档模型model对象，设置给Schema对象指定集合，并封装了增删改查的操作

       `let BookModel = mongoose.model('book', BookSchema)`

     - 增删改查 `userModel.method({}).then(function(data){})).catch(function(err){})`

       注意：Model.create() no longer accepts a callback，报错：

       ```js
       //报错
       BookModel.create({
           title: '西游记',
           author: '吴承恩',
           price: 19.9
       }, (err, data) =>{
           //错误
           console.log(err);
           //插入后的数据对象
           console.log(data);
       });
       
       //正确示范,then是成功插入数据库，catch是插入失败
       BookModel.create({
           title: '西游记',
           author: '吴承恩',
           price: 19.9
       }).then((data) =>{
               //插入后的数据对象
               console.log(data);
       }).catch((err) =>{
               //错误
               console.log(err);
       });
       ```

- 模块化

  完整流程：

  ```js
  //1. 安装 mongoose
  //2. 导入 mongoose
  const mongoose = require('mongoose');
  
  //设置 strictQuery 为 true
  mongoose.set('strictQuery', true);
  
  //3. 连接 mongodb 服务                        数据库的名称
  mongoose.connect('mongodb://127.0.0.1:27017/bilibili');
  
  //4. 设置回调
  // 设置连接成功的回调  once 一次   事件回调函数只执行一次
  mongoose.connection.once('open', () => {
    //5. 创建文档的结构对象
    //设置集合中文档的属性以及属性值的类型
    let BookSchema = new mongoose.Schema({
      name: String,
      author: String,
      price: Number,
      is_hot: Boolean
    });
  
    //6. 创建模型对象  对文档操作的封装对象    mongoose 会使用集合名称的复数, 创建集合
    let BookModel = mongoose.model('novel', BookSchema);
  
    //7. 新增
    BookModel.insertMany([{
      name: '西游记',
      author: '吴承恩',
      price: 19.9,
      is_hot: true
    }, {
      name: '红楼梦',
      author: '曹雪芹',
      price: 29.9,
      is_hot: true
    },]).then((err, data) => {
      //判断是否有错误
      if (err) {
        console.log(err);
        return;
      }
      //如果没有出错, 则输出插入后的文档对象
      console.log(data);
      //8. 关闭数据库连接 (项目运行过程中, 不会添加该代码)
      mongoose.disconnect();
    });
  
  });
  
  // 设置连接错误的回调
  mongoose.connection.on('error', () => {
    console.log('连接失败');
  });
  
  //设置连接关闭的回调
  mongoose.connection.on('close', () => {
    console.log('连接关闭');
  });
  
  
  ```

  

  - mongoose操作mongodb数据库的流程很清楚，但是代码耦合高，连接数据库，创建文档，文档指定集合，增删改查形成一个流程。

  - 一个普遍的需求：

    1. 连接数据库只写一次就行，复用
    2. 创建文档结构，指定集合可以根据逻辑修改，需要复用
    3. 增删改查也可以根据逻辑修改

  - 模块化的方法：

    1. 最重要也是唯一有用的武器是node自己的[模块方法](###2.3.1 模块化)，需要服用的代码封装暴露为一个函数：`module.exports=function(){}`
    2. 分析谁和谁解耦，或者说谁和谁分开
    3. 剪切需要复用或者逻辑修改的代码，放到一个新文件保存，并且module.exports暴露

  - 解耦连接数据库代码：

    1. `mongoose.connection.on('open', callbackFunction)`

    2. 核心问题：`mongoose.connection.on('open',)`和`callbackFunction`如何分开
    3. 思路：`mongoose.connection.on('open',)`当做函数暴露，`callbackFunction`当做函数的参数

    新文件db/db.js

    ```js
    module.exports=function(success,erro){
        if (typeof error !=='function'){
            error = ()=>{
                console.log('连接失败')
            }
        }
    
        // 导入mongoose
        const mongoose = require('mongoose')
    
        mongoose.connect('mongodb://127.0.0.1:27017/bilibili')
    
        mongoose.connection.once('open', ()=>{
            success()
        })
    
        mongoose.connection.once('error', ()=>{
            error()
        })
    
        mongoose.connection.on('close', () => {
            console.log('连接关闭')
        })
    
    }
    ```

    调用文件index.js

    ```js
    // 导入mongoose
    const mongoose = require('mongoose')
    
    // 导入连接数据库函数
    const db = require('./db/db')
    
    function updateDB(){
        let BookSchema = new mongoose.Schema({
            title: String,
            author: String,
            price: Number
        })
    
        let BookModel = mongoose.model('book', BookSchema)
    
        BookModel.create({
            title: '红楼梦',
            author: '曹雪芹',
            price: 19.9
        }).then(
            (err, data) =>{
                //错误
                console.log(err);
                //插入后的数据对象
                console.log(data);
        });
      }
    //调用函数
    db(updateDB)
    ```

  - 解耦创建文档结构、给文档指定集合和增删改查

    1. `let BookSchema = new mongoose.Schema();let BookModel = mongoose.model('book', BookSchema)`和`BookModel.create().then()`分开
    2. 需要明确哪两部分代码解耦，在结合`module.exports`语法思考如何封装。
    3. 再需要明确谁复用就封装谁

    新文件：model/BookModel.js

    ```js
    // 导入mongoose
    const mongoose = require('mongoose')
    
    let BookSchema = new mongoose.Schema({
        title: String,
        author: String,
        price: Number
    })
    
    let BookModel = mongoose.model('book', BookSchema)
    
    module.exports=BookModel
    ```

    调用文件index.js

    ```js
    // 导入mongoose
    const mongoose = require('mongoose')
    
    // 导入连接数据库函数
    const db = require('./db/db')
    const BookModel=require('./model/BookModel')
    
    function updateDB(){
        BookModel.create({
            title: '红楼梦',
            author: '曹雪芹',
            price: 19.9
        }).then(
            (err, data) =>{
                //错误
                console.log(err);
                //插入后的数据对象
                console.log(data);
        });
      }
    
    // 调用
    db(updateDB)
    
    ```

    

## 4.5 API接口

- 概述： API (Application Program Interface)，所以有时也称之为 API 接口

- 分类：

  - 软件开发工具包sdk（ Software Development Kit）中实现某种功能的函数
  - 以网络请求为基础的API

- 包API

  - 代码是在本地，调用封装的函数或者对象就可以实现API功能

  - 不需要经过网络请求

- 网络API

  - 代码不在本地，而是提供API的人将代码运行在服务器，通过网络请求提供调用

  - 前后端（客户端和服务端）通信的桥梁

  - 跟http协议和路由密不可分
  - 简单理解：一个接口就是服务中的一个路由规则+请求+响应结果
  - 接口组成：
    1. 请求方法
    2. 接口地址（URL）
    3. 请求参数
    4. 响应结果

### 4.5.1 RESTful API

[RESTful API](https://restfulapi.cn/)

- REST 是 Representational State Transfer 的缩写，如果一个架构符合 REST 原则，就称它为 RESTful 架构

- 背景

  - 客户端应用层出不穷，手机ios、安卓、电脑web，桌面软件、小程序等，客户端和服务端通信的接口必须统一，不然有很多重复性工作

  - RESTful API就是一套结构清晰、符合标准、易于理解、扩展方便让大部分人都能够理解接受的接口风格

![RESTful API](..\实例图片\RESTful API.jpg)

- 特点

  1. URL 中的路径表示资源 ，路径中不能有动词 ，例如 create , delete , update 等这些都不能有

  2. 操作资源要与 HTTP 请求方法对应

     - HTTP 请求方法：GET、POST、PATCH、DELETE对应资源的获取、新增、修改、删除
     - HTTP 请求方法不止这四种方法，还有[更多](https://www.runoob.com/http/http-methods.html)

     | HTTP协议请求方法 | RESTful API资源交互方式 | 描述                                                         |
     | ---------------- | ----------------------- | ------------------------------------------------------------ |
     | get              | 获取                    | `GET` 方法请求一个指定资源的表示形式，使用 `GET` 的请求应该只被用于获取数据。 |
     | post             | 新增                    | 向指定资源提交数据进行处理请求（例如提交表单或者上传文件）。数据被包含在请求体中。POST 请求可能会导致新的资源的建立和/或已有资源的修改。 |
     | patch            | 修改                    | `PATCH` 方法用于对资源应用部分修改。                         |
     | delete           | 删除                    | `DELETE` 方法删除指定的资源。                                |
     | put              |                         | 从客户端向服务器传送的数据取代指定的文档的内容。             |
     | head             |                         | 类似于 GET 请求，只不过返回的响应中没有响应体，用于获取响应头。 |
     | connect          |                         | HTTP/1.1 协议中预留给能够将连接改为管道方式的代理服务器。`CONNECT` 方法建立一个到由目标资源标识的服务器的隧道。 |
     | options          |                         | `OPTIONS` 方法用于描述目标资源的通信选项。                   |
     | trace            |                         | 回显服务器收到的请求，主要用于测试或诊断。                   |

  3. 操作结果要与 HTTP 响应状态码对应

  4. 返回结果是json格式

### 4.5.2 API测试图形工具：apifox

- 这种工具总共有三种
  - apifox https://www.apifox.cn/ (中文)
  - apipost https://www.apipost.cn/ (中文)
  - postman https://www.postman.com/ (英文)
- 背景
  - 现在有一个API，自己写的或者别人给我的。怎么快速测试功能呢？
  - get请求可以用浏览器，post，delete和patch呢？
  - 如果请求里还需要设置请求头和请求体呢？
  - 那么就需要一个api测试工具了
- 使用
  1. 设置请求方法，url（里面可能有query参数和params参数）
  2. 如果有必要，还可以设置请求头和请求体
  3. 之后就可以得到相应结果，相应结果里包括响应头和响应体
  4. ==图形工具还有一个功能，生成接口文档，在线文档，在线运行、调试==
- [apifox很强大](开发辅助工具)，[视频教学](https://www.bilibili.com/video/BV1ae4y1y7bf/?vd_source=5a1ce44ffe3923badc10b4b00217e698)

### 4.5.3 API测试命令行工具：cURL

- 概述

  - client url
  - 一个计算机命令行工具，通过命令行与服务器进行数据交互
  - 可以做的事：文件上传、下载，获取信息
  - 支持很多协议：例如HTTP、HTTPS、FTP等
  - curl是一个跨平台的工具，你可以在各种操作系统上使用它，如Windows、Mac和Linux等
  - curl提供了很多参数和选项，可以根据需要进行自定义配置。例如，你可以指定请求方法、请求头、请求体和验证信息等。curl还支持cookie管理、代理设置以及上传和下载文件的功能

- 必会命令

  1. 基本框架请求：`curl URL`

     - 例如：`curl baidu.com`

     - 默认get请求
     - 结果就是API的返回资源

  2. 请求方法选项：`cur -XPOST URL`或者`cur -X -POST URL`

  3. 请求头： `curl -H {'xx':'yy','ee':'dd'} URL`或者 `curl -H 'Content-Type: application/json' URL`

  4. 发送 POST 请求的数据体：`curl -d 'login=emma＆password=123' URL`

  5. HTTP 请求跟随服务器的重定向：`curl -L URL`

- 注意：curl请求中query参数是?cc=yy&rr=xx，往往出错，因为这里的&是特殊字符，需要处理，比较好的办法是双引号框住url，例如：

  `curl "https://api.github.com/search/users?q=babyfox&per_page=2"`

- [更多参数选项参考](https://zhuanlan.zhihu.com/p/336945420)

### 4.5.4 RESTful API编程思路

- 本质还是路由规则+资源，只不过对方法和路径有限制，返回结果有格式限制
- 请求方法：
  - get：获取资源
  - post：新增资源
  - patch：修改资源
  - delete：删除资源
- 返回json数据，node中是：`res.json({...})`，json中的字段一般是：
  - status
  - msg
  - data

## 4.6 会话控制

- 背景

  1. 按照http协议这套逻辑（请求报文和响应报文），客户端和服务端的通信，很完美的解决了

     **Q：但后来还是出现了一个需求：服务器需要根据不同的客户端使用者，提供不同的读写权限，提供不同的内容。该怎么分辨用户？**

     **A：直觉来看，这需求很简单，服务器创建一个用户数据库，客户端发请求，在请求头或者什么地方带上用户名和密码，服务端收到数据检验一下，提供不同的读写权限，提供不同的内容。**

  2. 客户端向服务端发请求很多很频繁，客户端一个操作至少一个请求，甚至一个操作不止一个请求

  3. http协议有个特征：无状态，换言之，前后两个请求是没有联系的。

  4. http协议只规定客户端-服务器最基本的通信：

     - 请求方式，url，请求头和请求体

       客户端都可以url，请求头和请求体里夹带私货给服务端传东西

     - 响应头，响应体

       服务端也可以在响应头，响应体里给客户端传东西

     **Q：虽然服务器的数据库可以分辨用户了，但是有一个新问题：，每一次请求都需要输入密码，用户体验很差，如何做到输入一次用户名密码，之后服务器记住用户？**

     **A：继续用直觉思考，用户客户端第一次输入用户名密码通过服务器的校验，服务器在响应报文里给客户端什么东西，客户端有这个东西，后面每一次给服务器发请求带上这个东西，就不用输入密码了。这样既解决了分辨用户，还可以让服务器记住用户。第一代技术**

  5. 上面说到服务器校验客户端登陆数据后，响应客户端一个东西，客户端之后每次请求带上这个东西。辨别用户，记住用户。**这个东西就是cookie，存储在客户端浏览器**。

  6. cookie被发明出来，其实不仅仅是辨别用户，还能记录用户行为等等。人们开发 Cookie 的目的是使用户和网站之间的互动更加便捷。如果没有 Cookie ，网站将无法记录用户偏好，登录信息，也无法提供[购物车功能](https://www.zhihu.com/search?q=购物车功能&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1548191596})。这意味用户每次访问淘宝，微博，豆瓣等网站时都要重新登录，而失去购物车功能会令网购体验非常糟糕。参考：

     - [大厂是如何在网上用Cookie跟踪你的](https://www.bilibili.com/video/BV1Hi4y1x7nP/?spm_id_from=333.337.search-card.all.click&vd_source=5a1ce44ffe3923badc10b4b00217e698)
     - [谷歌禁用第三方cookie，为了保护你的隐私，还是为了更好的垄断](https://www.bilibili.com/video/BV1ki4y1P7ya/?spm_id_from=333.337.search-card.all.click&vd_source=5a1ce44ffe3923badc10b4b00217e698)
     - [Cookies如何记录隐私信息？](https://www.zhihu.com/question/274363469)
     - [ETP、ITP、NO-TP，是时候把第三方Cookie讲清楚了](https://maxket.com/wtf-is-3rd-party-cookie/)

     **Q：cookie可以做的事情真的挺多，辨别用户只是其中之一，微乎其微。有的甚至还有记录你购物车里是什么，所以又有新问题了：cookie这东西是存储在浏览器，它里面一对key-value值记录一个信息，直接明文的话，会不会不安全？cookie可以干的事情那么多，cookie就很多啊，都存在浏览器，浏览器会不会不堪其重？**。

     **A：继续直觉思考，既然浏览器压力大，cookie又挺多，那把这些key-value信息给服务器存储吧。按照这思路，从登陆开始捋一捋，服务器校验完了，把这些key-value信息存起来，那还是得给客户端什么东西啊，不然下次客户端怎么无密码请求？服务器响应客户端啥呢？key-value保存在服务器，不可能把key-value响应吧，那这个思路还是之前的方案。那不如把key-value的索引响应给客户端，客户端的cookie里只是一个个索引，安全，存储小。但是，客户端之后每次请求cookie里带上这个索引，服务器通过索引就能找到真正的key-value信息。好，解决了客户端cookie明文安全，存储压力大的问题，把这个问题转移给了服务器。第二代技术session**

  7. session把数据key-value从浏览器转移到服务器，服务器返回到浏览器的是一个索引，实际的信息存在服务器的内存或者数据库

     **Q：但很快又有了新问题，现在网络引用越来越复杂，用户相关的数据越来越多，比如购物车内容，用户喜好设置，用户浏览行为等，都存在服务器的session里，给客户端浏览器返回索引，客户端解放了，服务器端压力很大，尤其是用户数量还很多的时候，怎么办？怎么让服务器的存储和请求压力小一点？**

     **A：服务器压力大，那要不还是保存在浏览器吧，那样又会出现cookie的问题，cookie里的name和value字段存储能力有限，而且还是明文。那就想个办法，对cookie里的name和value进行压缩编码。第三代技术token，其中压缩编码的技术是JWT**

  8. JWT是实现token技术的根本，本质是压缩编码。原理参考：

     [JWT详细讲解](https://developer.aliyun.com/article/995894)

- 关于cookie的实验：

  1. 打开[淘宝](https://www.taobao.com/)，尚未登陆。
  2. 禁止所有cookie（==所谓禁止cookie，就是服务器响应的时候会返回cookie，浏览器下次请求会自动把cookie放到请求头，这个行为是浏览器完成的，禁止cookie是浏览器不干这件事了，所以浏览器其实是收到cookie的，也存下来了，但是不在给服务器==），这时候淘宝网页都不完整，看不到商品。无法点击。
     - F12，发现在应用-cookie里是有东西的，ctrl+r刷新，cookie不会变化
     - 在元素里，找到淘宝商品对应的html，发现只有样式，没有链接
  3. 开启cookie，还是不登录，但是淘宝网页完整了。
     - F12，发现在应用-cookie里是有东西的，ctrl+r刷新，cookie会变化了
     - 在元素里，找到淘宝商品对应的html，发现样式链接什么都有了
  4. 禁止第三个cookie，不登录，网页还是完整。
     - F12，发现在应用-cookie里是有东西的，ctrl+r刷新，cookie会变化了
     - 同时，cookie里会多一个https://g.alicdn.com/，里面的内容和cookie里的https://www.taobao.com/相同。注意，前面说的cookie是指cookie里的https://www.taobao.com/，https://g.alicdn.com/里啥也没有，但现在禁用第三方cookie以后，出现了这个https://g.alicdn.com/，里面的cookie和https://www.taobao.com/里面的一样。
  5. 开启所有的cookie，登陆。
     - F12，发现在应用-cookie里是有东西的，多了很多东西

- 个人发现：

  - 网站很多内容需要服务器读取客户端的cookie才会响应，即便是并没有登陆
  - cookie绝对不是用来专门解决辨别用户和记住用户的。
  - cookie本质是请求头里一个存储用户信息的东西
  - ~~从你打开淘宝网站，就会自动把一些cookie装到请求头，如果不让服务器读你的cookie，那么服务器可以不给你响应最基本的内容~~
  - 打开淘宝网站，服务器可以直接给你发送cookie，但是把这个cookie自动放在以后每次请求头里是浏览器做的，可以设置浏览器不这么干，服务器没收到你的cookie，可以不给你响应最基本的内容。

- 问题1：假如我是服务器，我该怎么让客户端把它的隐私信息放到cookie里？

- 问题2：假如我是互联网用户，通过浏览器等客户端访问服务器里的内容，我也登陆了用户名密码，服务器就知道了我是谁。按道理说，我在网站的行为，服务器应该很清楚啊，服务器大可以创建一个数据库，用一些字段记录我的行为。干嘛非要通过cookie获取我的行为信息？

### 4.6.1 cookie技术

- cookie字段含义

  | 字段名          | 作用                                                         |
  | :-------------- | :----------------------------------------------------------- |
  | name            | cookie中信息的key，服务器自己定义。                          |
  | value           | cookie中信息的value，还是服务器自定义。如何没有session，value里存储的就是信息，比如邮箱地址等。如何有session，信息存在服务器里的session，这时候，服务器会定义cookie里的value是session信息的索引，也就是sessionId。 |
  | domain          | 为可以访问此cookie的域名，顶级域名只能设置domain为顶级域名，不能设置为二级域名或者三级域名，否则cookie无法生成。二级域名能读取设置了domain为顶级域名或者自身的cookie，不能读取其他二级域名domain的cookie。所以要想cookie在多个二级域名中共享，需要设置domain为顶级域名，这样就可以在所有二级域名里面或者到这个cookie的值了。 |
  | path            | 为可以访问此cookie的页面路径。 比如domain是abc.com,path是/test，那么只有/test路径下的页面可以读取此cookie。 |
  | expires/Max-Age | 设置cookie的过期时间。不设置的话默认值是Session，意思是cookie会和session一起失效。当浏览器关闭(不是浏览器标签页，而是整个浏览器) 后，此cookie失效。 |
  | size            | 设置cookie的大小                                             |
  | httpOnly        | cookie的httpOnly属性。若为true，则只有在http请求头中会带有此cookie的信息，而不能通过document.cookie来访问此cookie |
  | secure          | 设置是否只能通过https来传递此条cookie                        |
  | 等等其他        |                                                              |

- 编程思路

  1. 安装`cookie-parser`中间件：`npm i cookie-parser`

  2. 导入`cookie-parser`，引入中间件

     ```js
     //导入 express
     const express = require('express');
     //导入`cookie-parser`
     const cookieParser = require('cookie-parser')
     
     //创建应用对象
     const app = express();
     //引入中间件cookie-parser
     app.use(cookieParser());
     ```

  3. `res.cookie()`设置服务器返回的cookie

  ```js
  //导入 express
  const express = require('express');
  const cookieParser = require('cookie-parser')
  
  //创建应用对象
  const app = express();
  app.use(cookieParser());
  
  //创建路由规则
  app.get('/set-cookie', (req, res) => {
    res.cookie('name', 'zhangsan'); // 会在浏览器关闭的时候, 销毁
    res.cookie('email','271293513@qq.com', {maxAge: 60 * 1000}) // max 最大  age 年龄
    res.cookie('theme', 'blue');
    res.send('home');
  });
  
  //删除 cookie
  app.get('/remove-cookie', (req, res) => {
    //调用方法
    res.clearCookie('name');
    res.send('删除成功~~');
  });
  
  //获取 cookie
  app.get('/get-cookie', (req, res) => {
    //获取 cookie
    console.log(req.cookies);
    res.send(`欢迎您 ${req.cookies.name}`);
  })
  
  //启动服务
  app.listen(3000);
  ```

- cookie-parser中间件

  1. 安装，导入，引入为全局中间件

  2. 最重要的api：`res.cookie(name, value [, options])`，其中的参数正好对应保存在浏览器cookie的字段

     - name：对应name

     - value：对应value

     - option: 类型为对象，可使用的属性对应那些字段，如下

       ![cookie-parser](..\实例图片\cookie-parser.png)

- 关于签名   `res.cookie(name,value,{signed:true})`

  - 一个基本事情是：cookie里的信息只是存在name和value中，浏览器里保存的cookie很多字段，除了name和value以外，其他字段只是cookie的机制，用来描述name和value的性质，不让访问域名，路径，生命周期等。

  - 其中一个重要的性质就是signed，是否需要签名

  - cookie存储在浏览器，用户可以自己篡改的。然后cookie里的信息是关于用户的，通过篡改cookie，有没有可能伪装成谁去和服务器通信？

  - 再复述客户端和服务端通信过程：客户端浏览器发请求访问，需要输入用户名密码，服务器收到请求，校验用户名和密码，然后在响应里返回用户name和value，还有一些规定name和value性质的选项。浏览器会先保存收到的cookie，里面的数据是name：user，value：zhangsan。**假如张三想伪装成李四跟服务器交互，他不知道李四的密码，用户名倒是知道，直接通过登陆页面输入用户名密码肯定不行。还有一种办法，cookie可以让用户输入一次用户名密码，之后请求时带上cookie就可以不用登陆另外。拿到李四的cookie就行。但是他的cookie在他的浏览器上，拿不到啊。于是开始研究自己的cookie，发现cookie里name和value好像就是用户名啊，那我何不改我自己的cookie，让服务器以为我是李四。试一下还真成功了。**

  - 为了解决篡改cookie，伪装别人的问题，出现了签名这个技术。

  - 签名其实就是，对某一个信息进行加密之前，加上一个只有我才知道的密钥， 对数据做一个签名

    ![签名](..\实例图片\签名.png)

- 签名编程：cookie里的value值，经过加密，防止修改，额外操作

  1. 设置密钥：`app.use(cookieParser('babynose'))`
  2. 设置cookie，加上选项：` res.cookie('theme', 'blue',{signed:true})`
  3. 通过`req.signedCookies`获取

  ```js
  //导入 express
  const express = require('express');
  const cookieParser = require('cookie-parser')
  
  //创建应用对象
  const app = express();
  //设置密钥
  app.use(cookieParser('babynose'));
  
  //创建路由规则
  app.get('/set-cookie', (req, res) => {
    res.cookie('name', 'zhangsan'); // 会在浏览器关闭的时候, 销毁
    res.cookie('theme', 'blue',{signed:true});
    res.send('home');
  });
  
  //获取 cookie
  app.get('/get-cookie', (req, res) => {
    //获取 cookie
    console.log(req.cookies.name) //zhangsan
    console.log(req.secret) //babynose
    console.log(req.signedCookies.theme) //blue
    res.send(`欢迎您:${req.cookies.name}, theme:${req.signedCookies.theme}`);
  })
  
  //启动服务
  app.listen(3000);
  ```

### 4.6.2 session技术

- 理解session的基本要点：

  - session存储在服务器，数据库或者内存
  - session本身也有机制，或者说session信息的选项，类似于cookie的那些非name和value字段
  - session的思想是用户数据存在数据库，给cookie返回一个索引，也就是说，cookie里的那些字段，原本就是服务器指定的，一切不变，只是value的值不是原来的明文或者加密的，而是一个索引，真正的数据转移到服务器了。

- 编程思路

  1. 安装`express-session`中间件操作session : `npm i express-session`

  2. 安装`connect-mongo`中间件 连接数据库: `npm i express-session`

  3. 导入两个中间件

  4. 配置中间件

     ```js
     // 导入session中间件
     const session = require('express-session')
     const MongoStore = require('connect-mongo')
     
     //配置session中间件，session数据保存数据库
     sessionMiddleware = session({
       name: 'sid',   //设置cookie的name字段，默认值是：connect.sid
       secret: 'babynose', //密钥，参与加密的字符串（又称签名）  加盐
       saveUninitialized: false, //是否为每次请求都设置一个cookie用来存储session的id
       resave: true,  //是否在每次请求时重新保存session  20 分钟    4:00  4:20
       store: MongoStore.create({
         mongoUrl: 'mongodb://127.0.0.1:27017/accounts' //数据库的连接配置,mongodb是协议
       }),
       cookie: {
         httpOnly: true, // 开启后前端无法通过 JS 操作
         maxAge: 1000 * 60 * 60 * 24 * 7 // 这一条 是控制 sessionID 的过期时间的！！！
       },
     })
     
     ```

     session中间件里的配置很全面：

     - cookie：里面正是浏览器里cookie字段，不包括name和value，这里的配置的是cookie的性质
     - name：cookie里的name。cookie的value不需要指定，这是seesion的规定死的机制，value是session数据的索引。
     - secret：密钥，对value，也就是sessionId加盐，防止篡改
     - store：session里真正的数据存在哪里

     [深入浅出 Express 中间件 Express-session)](https://zhuanlan.zhihu.com/p/409813376)

  5. session中间件放到路由中，给session添加真正的数据。（==当然可以设置为全局中间件，大多数情况是全局中间件==）

     ```js
     // 登陆，服务端处理
     router.post('/login',sessionMiddleware,(req,res,next)=>{
         console.log(req.body)
         let {username,password}=req.body
     
         // 验证用户名密码
         UsersModel.findOne({username:username,password:md5(password)})
         .catch(
             (err)=>{
                 res.status(500).send('登陆失败，请稍后再试')
             }
         )
         .then(
             (data)=>{
                 //密码错误，data是null
                 if (!data){
                     return res.send('账号或密码错误')
                 }
                 //写入session
                 //这里有两个数据集，users和sessions
                 //data是数据库中的users集合里的数据
                 //req.session.username是写到了sessions数据库，但是返回到cookie的是索引
                 req.session.username=data.username
                 req.session._id = data._id
                 req.session.password=data.password
     
                 //登陆成功
                 res.render('success',{msg:'登陆成功',url:'/account'})
             }
         )
     })
     ```

     - `req.session.username = data.username`其中，`req.session.username`是`session`中的字段`username`，`data.username`是从服务器数据库里读取的数据
     - 一般情况有两个数据库：用户数据库，比如用户名密码。session数据库，里面是用户的行为数据或者其他的，`req.session.xx`设置字段保存什么数据。session数据保存位置是通过前中间件`session(...store:...)`配置

  6. session中间件放路由里，获取session里的数据：`req.session.xx`

     ```js
     // 跳转添加记录
     router.get('/account/create', checkLoginMiddleware,function(req, res, next) {
           if(!req.session.username){
                 return res.redirect('/login')
             }
         res.render('create');
     });
     ```

     

- 关于表单form

  1. form表单里面的input标签加入name属性，必须和数据库字段相同
  2. form表单method属性添加http协议请求方法，action属性添加url
  3. 根据form里的方法和url添加路由规则

### 4.6.3 token技术

- 特点

  - 用户数据存浏览器，但是不在cookie。也不在服务端的session
  - 用户数据虽然在浏览器，但是进行了压缩编码，很小的存储，但是信息却可以存储很多
  - token技术需要手动将token数据添加在请求报文中，一般是放在请求头中。cookie和session技术都是浏览器来自动将cookie数据添加到请求头
  - JWT，JSON Web Token，是实现token技术的根本，JWT 使 token 的生成与校验更规范

- 编程思路

  [API参考]([jsonwebtoken - npm (npmjs.com)](https://www.npmjs.com/package/jsonwebtoken))

  1. 安装`JWT`：`npm i jsonwebtoken`

  2. 导入token：`const jwt = require('jsonwebtoken') `

  3. 编码token：`jwt.sign(payload, secretOrPrivateKey, [options, callback])`

     - `playload`：编码的数据，可以是表示有效 JSON 的对象文本、buffer，或字符串。

     - `secretOrPrivateKey`：密钥，加盐操作，防止数据太简单被破解

     - `options`：配置对象，配置选项如下：

       - `algorithm`：编码算法，默认值**HS256**

       - `expiresIn`：token有效时间，单位是秒

       - `notBefore`

       - `audience`

         ...

     - callback：异步的回调函数

  4. 解码token：`jwt.verify(token, secretOrPublicKey, [options, callback])`

     - `token`：编码的token

     - `secretOrPublicKey`：密钥

     - `options`：同样是配置对象，配置选项如下：

       - `algorithm`s：编码算法，如果未指定，将根据提供的密钥类型使用默认值

       - `maxAge`：令牌仍然有效的最长允许期限

       - `audience`

         ...

- `playload`里面有规则

  1. jwt基本知识：`JWT`是`token`的一种形式。主要由`header（头部）`、`payload（载荷）`、`signature（签名）`这三部分字符串组成，这三部分使用"."进行连接，完整的一条`JWT`值为`${header}.${payload}.${signature}`

  2. `header`最开始是一个`JSON`对象，该`JSON`包含`alg`和`typ`这两个属性。之后会对`JSON`使用`base64url`（使用`base64`转码后再对特殊字符进行处理的编码算法）编码后得到的字符串就是`header`的值。

     ```json
     {
       "alg": "HS256",
       "typ": "JWT"
     }
     
     注释：
     alg：签名算法类型，生成JWT中的signature部分时需要使用到，默认HS256，在jsonwebtoken包里是通过algorithm指定
     typ：当前token类型
     ```

  3. 关键的`playload`，跟`header`一样,也是一个`JSON`对象，使用`base64url`编码后的字符串就是最终的值。只不过`payload`中存放着7个官方定义的属性，也就是Registered claims 。同时我们可以写入一些额外的信息，例如用户的信息等，也就是Public claims。官方定义的7个推荐属性如下：

     ```json
     {
         iss：签发人
         sub：主题
         aud：受众
         exp：过期时间
         nbf：生效时间
         iat：签发时间
         jti：编号
     }
     ```

  4. `signature`会使用`header`中`alg`属性定义的签名算法，对`header`和`payload`合并的字符串进行加密，加密过程伪代码如下：

     ```js
     HMACSHA256(
       `${base64UrlEncode(header)}.${base64UrlEncode(payload)}`,
       secret
     )
     
     注释：
     signature实际就是jsonwebtoken包里是通过secretOrPublicKey指定
     ```

## 4.7 部署

- [VScode 代码上传仓库（码云，github）](#8. VScode 代码上传仓库（码云，github）)



# 5 程序概念

## 5.1 [路由管理](####2.4.2.2 路由规则管理)

## 5.2 [中间件](####2.4.3.3 中间件)

## 5.3 [模板引擎](####2.4.3.4 模板引擎)

## 5.4 CSRF跨站伪造

- 概述
  - CSRF，Cross-site request forgery，跨站请求伪造
  - 利用受害者尚未失效的身份认证信息（cookie、会话等），诱骗其点击恶意链接或者访问包含攻击代码的页面，在受害人不知情的情况下以受害者的身份向（身份认证信息所对应的）服务器发送请求，从而完成非法操作（如转账、改密等）。
  - CSRF与XSS最大的区别就在于，CSRF并没有盗取cookie而是直接利用。

- 思路：

  - 用户访问网站A，它是存在于服务器A的，在浏览器上有服务器A响应的cookie

  - 伪造者用服务器B，创建了一个网站B。用户访问网站B里的某些操作，实际是会携带coookie去访问网站A完成一些操作。这就是CSRF攻击，做到这些，需要条件：
    1. 很清楚服务器A里的一些路由规则对应的操作，这样才能去伪造。需要详细研究网站A的漏洞。
    2. 用户在网站B，跟服务器B交互，交互的过程中去想服务器A发送请求，却能拿到网站A的cookie
    3. 把网站B送达用户

- 举例：

  1. A服务器的API：http://127.0.0.1:3000/accounts?id=1 是获取id为1的详细账单信息
  2. 本来用户需要用户名密码登陆才有资格获取权限请求，但是第一次输入登陆以后，服务器给了cookie，下次就不用了，cookie保存在浏览器，浏览器会在下次访问时把cookie自动放在请求头中，就能获取权限。
  3. 这时候，服务器B开发了网站B，里面的a标签，form标签等可以发出url请求，但是请求不是给服务器B自己，而是一个站外请求，所以这也是为什么叫跨站伪装。站外请求url是指向服务器A，与此同时，有一种机制就是站外请求浏览器也会自动在请求头中加入cookie，所以站外请求也能获取权限。更不幸的是，网站B发出站外请求，不一定需要用户点击，比如link标签就和a标签不一样，自动发出请求，服务器B就可以通过这样的方式，悄无声息的替你和服务器A交互。
  4. 总结，服务器B可以访问服务器A，浏览器会自动填充服务器A的cookie。并且这个访问请求可以悄无声息。

# 3. ajax

# 4. typescript

# 5. npm

## 5.1[关联](##3.2包管理 "回顾开发生态")

## 5.2概述

- Node Package Manager 『node的包管理工具』
- 安装node.js会自动安装npm，通过**npm -v**查询版本

## 5.3命令

- 初始化

  ```cmd
  cd dir
  npm init
  ```

  1. 这是一个交互式的命令，运行后会有一系列问题

  2. 以某目录（文件夹）作为工作目录，执行 **npm init**，会将此文件夹初始化为一个『包』，生成**package.json**文件

  3. 重要的是，**package.json**里面的**main**配置好了： "main": "index.js"

     ```json
     {
     "name": "1-npm", #包的名字
     "version": "1.0.0", #包的版本
     "description": "", #包的描述
     "main": "index.js", #包的入口文件
     "scripts": { #脚本配置
     "test": "echo \"Error: no test specified\" && exit 1"
     },
     "author": "", #作者
     "license": "ISC" #开源证书
     }
     
     ```

     

  4. 当然index.js需要自己写

- 找到自己需要的包

  - https://www.npmjs.com/

- 下载安装

  ```cmd
  # 格式
  npm install <包名>
  npm i <包名>
  
  # 示例
  npm install uniq
  npm i uniq
  ```

  1. 全局安装下载

     ```cmd
     # 安装 vue-cli
     npm install -g @vue/cli 
     
     # 安装 nodemon
     npm i -g nodemon
     ```

     - 全局安装的npm包跟局部安装的不一样，局部npm包需要在main,js文件中通过require导入，全局npm包是通过独立命令使用，在命令行使用。

       ```cmd
       #vue创建工程
       vue create my_project
       
       # nodemon代替node动态的处理代码main.js变化
       nodemon ./main.js
       ```

     - 全局npm包和局部npm包，安装位置也不一样。局部npm包安装在当前目录下的**node_modules** 文件夹下。全局npm包安装在C:\Users\admin\AppData\Roaming\npm\node_modules

     - 不是所有的包都适合全局安装 ， 只有全局类的工具才适合，可以通过 查看包的官方文档来确定 安装方式 

       ==其实**vue-cli（vue脚手架）**是node.js上的一个全局npm包==

  2. 在文件夹dir下，运行npm i 命令发生了什么，生成两个资源：

     - **node_modules** 文件夹，存放下载的包
     - **package-lock.json** 包的锁文件 ，用来锁定包的版本

- 安装依赖：假如项目里没有**node_modules** 文件夹，该怎么办？

  ```cmd
  npm i
  # 或者
  npm install
  ```

  [点击跳转详细介绍](###5.5.2 构建环境)

- 安装指定版本的包和删除包

  ```cmd
  ## 格式
  npm i <包名@版本号>
  ## 示例
  npm i jquery@1.11.2
  
  ## 局部删除
  npm remove uniq
  npm r uniq
  ## 全局删除
  npm remove -g nodemon
  ```

- 配置命令别名：node命令可能参数很多，有更好的办法吗？

  ```cmd
  # 预设 server,build,start命令
  npm run server
  
  npm run build
  
  npm start
  ```

  [哪里预设命令](###5.2.3 配置预设命令)

- 配置淘宝镜像 

  `npm config set registry https://registry.npmmirror.com/`

- 发布、更新、删除自己的npm包

  1. 某目录下初始化，`npm init` ，命令行会提示输入包名字，除了注意不要大写和中文本身的规则，还不要带有test和learn等，因为官方有垃圾检测机制，发布的包可能不会通过

  2. 创建index.js文件，在文件中声明函数，使用 module.exports 暴露

     ```javascript
     //加法
     function add(a,b){
       return a + b;
     }
     
     //减法
     function sub(a, b){
       return a - b;
     }
     
     //暴露
     module.exports = {
       add,
       sub
     }
     ```

  3. 注册账号 https://www.npmjs.com/signup ，激活

  4. 修改包管理镜像，因为大多数情况，国内会使用淘宝镜像，速度快，但是淘宝镜像是只读镜像，只能下载，不能上传包。

     ```cmd
     # 安装 nrm，一个全局npm包，一个管理npm镜像的工具
     npm i -g nrm
     
     # 修改为淘宝镜像，这里不需要
     nrm use taobao
     
     # 修改为npm镜像
     nrm use npm
     ```

  5. 登陆：`npm login`  填写注册的用户名和密码

  6. 发布：`npm publish` ，就可以下载使用了

  7. 更新

     - 更新index.js代码
     - 测试是否可用，一般会新建一个test.js来测试
     - 修改 package.json 中"version": "1.0.1"配置，指定版本号
     - 发布：`npm publish`

  8. 删除：`npm unpublish --force`

     删除包需要满足一定的条件，https://docs.npmjs.com/policies/unpublish 

     - 你是包的作者 
     - 发布小于 24 小时 
     - 如果大于 24 小时后，没有其他包依赖，并且每周小于 300 下载量，并且只有一个维护者

## 5.4 require导入npm包的流程

- 当前目录（文件夹）dir下，main.js导入npm包，不需要路径，直接写名字

  ```javascript
  const uniq = require('uniq');
  ```

- 那么，会在当前文件夹dir下 **node_modules**文件夹中寻找同名的uniq文件夹

  - 其实uniq文件夹里面会有具体实现功能函数的js文件，以及**package.json**文件以及里面的**mian**配置
  - 换句话说，npm包本质还是当做node.js的自定义文件

- 如果当前目录下没有node_modules文件夹，在上级目录中下的 node_modules 中寻找同名的文件夹，直至找到磁盘根目录

## 5.5 package.json和package-lock.json

### 5.5.1 概述

- package-lock.json 文件
  1. 使用 `npm install` 命令安装包后就会自动生成
  2. 但是`package-lock.json`并不是每次都生成，只有在没有的情况下才会自动生成。
  3. 当存在并且有包的变化的时候会自动同步更新。
- package.json 文件
  1. 初始化命令 `npm init`生成
  2. 有包的变化的时候也会自动同步更新，
  3. 也就是说，运行`npm i uniq`，package.json和package-lock.json都会更新

### 5.5.2 构建环境

- 项目里没有**node_modules** 文件夹里的依赖环境
  1. 这种情况很普遍，因为node_modules里的依赖环境本身只是引用，并不是项目里的成果。并且node_modules里面的空间往往很大。
  2. 克隆项目的时候一般是没有**node_modules** ，但是有package.json和package-lock.json文件
  3. 终端运行命令`npm install`会创建node_modules文件夹，并安装依赖。那么，`npm install`命令是怎么做到的？
     - package.json和package-lock.json有两个配置：dependencies和devDependencies，生产依赖和开发依赖。
     - 生成依赖树：首先会检查下项目中是否有 `package-lock.json` 文件：存在 `lock` 文件的话，会判断 `lock` 文件和 `package.json` 中使用的依赖版本是否一致，如果一致的话，就使用 `lock` 中的信息。反之就会使用 `package.json` 中的信息；那如果没有 `lock` 文件的话，就会直接使用 `package.json` 中的信息生成依赖树。
     - 将 `npm` 远程仓库中的包下载至本地

### 5.2.3 配置预设命令

- 配置 package.json 中的 scripts 属性

  ```json
  {
  .
  .
  .
  "scripts": {
  "server": "node server.js",
  "start": "node index.js",
  },
  .
  .
  }
  
  ```

- 那么，`npm run server` = `node server.js`

- 不过 start 别名比较特别，使用时可以省略 run

  `npm start` = `node index.js`

### 5.2.4 配置详解

![package_json](C:\Users\admin\Desktop\coolShare\assert\package_json.png "package.json配置")

[参考](https://juejin.cn/post/7137110113277444126))

## 5.6包管理工具汇总

|    语言    |     包管理工具      |
| :--------: | :-----------------: |
|    PHP     |       compose       |
|   Python   |         pip         |
|    Java    |        maven        |
|     Go     |       go mod        |
| JavaScript | npm/yarn/cnpm/other |
|    Ruby    |      RubyGems       |

- 除了编程语言领域有包管理工具之外，操作系统层面也存在包管理工具，不过这个包指的是『 软件包 』

| 操作系统 | 包管理工具 | 网址                                |
| :------: | :--------: | :---------------------------------- |
|  Centos  |    yum     | https://packages.debian.org/stable/ |
|  Ubuntu  |    apt     | https://packages.ubuntu.com/        |
|  MacOs   |  homebrew  | https://brew.sh/                    |
| Windows  | chocolatey | https://chocolatey.org/             |

# 6. vue-cli,vite,webpack

# 7. html,css,javascript

#  8. VScode 代码上传仓库（码云，github）

# 8.1 初始化仓库

1. 需要上传的文件夹，以此为根目录，用vscode打开

2. 点击导航栏的源代码管理，点击初始化仓库

3. 回到导航栏的资源管理器，创建文件`.gitignore`，编辑需要忽略的部分，或者说不上传到仓库的部分

   ```git
   node_modules
   ```

   [gitignore文件的配置使用](https://zhuanlan.zhihu.com/p/52885189)

# 8.2 创建本地仓库

1. 点击导航栏的源代码管理，输入init，提交
2. 出现错误

#  8.3创建远程仓库

1. 平台：github或者码云
2. 注册登陆
3. 创建空仓库：网站内操作，新建仓库-填写表单-创建
4. 复制仓库地址

# 8.4 本地仓库提交到远程仓库

1. 回到vscode，源代码管理-三个点-远程-添加远程存储库，粘贴远程仓库地址
2. 输入仓库名字
3. 输入账号密码
4. 发布到branch



