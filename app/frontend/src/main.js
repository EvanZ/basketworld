import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'
import { library } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { faRedo, faLocationArrow, faPaperPlane, faBullseye, faToggleOn, faToggleOff, faKeyboard } from '@fortawesome/free-solid-svg-icons'
import { faHandPointer } from '@fortawesome/free-regular-svg-icons'

library.add(faRedo, faLocationArrow, faPaperPlane, faBullseye, faHandPointer, faToggleOn, faToggleOff, faKeyboard)

const app = createApp(App)
app.component('font-awesome-icon', FontAwesomeIcon)
app.mount('#app')
